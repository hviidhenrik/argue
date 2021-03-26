import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model

tf.get_logger().setLevel('ERROR')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from src.utilities.utility_functions import *
from src.data.data_utils import *
from src.config.definitions import *
from src.models.AAE.aae_class import *
import time

from tensorflow_gan.python.losses.losses_impl import wasserstein_generator_loss, wasserstein_discriminator_loss


class DAD(AdversarialAutoencoder):
    def __init__(self,
                 autoencoder_latent_dimension: int = 2,
                 autoencoder_activation: Union[str, List[str]] = "elu",
                 encoder_hidden_layers: List[int] = None,
                 decoder_hidden_layers: List[int] = None,
                 discrim_hidden_layers: List[int] = None,
                 discrim_activation: Union[str, List[str]] = "elu",
                 anomaly_generator_input_size: int = 7,
                 anomaly_generator_hidden_layers=None,
                 anomaly_generator_activation_function: Union[str, List[str]] = "elu",
                 anomaly_generator_dropout_fraction: float = 0,
                 anomaly_generator_loss_function: tf.keras.losses = tf.keras.losses.MeanSquaredError(),
                 _anomaly_generator_optimizer: tf.keras.optimizers = tf.keras.optimizers.Adam(),
                 alarm_hidden_layers=None,
                 alarm_activation_function: Union[str, List[str]] = "elu",
                 alarm_dropout_fraction: float = 0,
                 alarm_loss_function: tf.keras.losses = tf.keras.losses.BinaryCrossentropy(from_logits=False),
                 _alarm_optimizer: tf.keras.optimizers = tf.keras.optimizers.Adam()
                 ):
        anomaly_generator_hidden_layers = [5, 4] if anomaly_generator_hidden_layers is None else \
            anomaly_generator_hidden_layers
        alarm_hidden_layers = [5, 4] if alarm_hidden_layers is None else alarm_hidden_layers
        self.autoencoder_latent_dimension = autoencoder_latent_dimension
        self.anomaly_generator_input_size = anomaly_generator_input_size
        self.anomaly_generator_hidden_layers = anomaly_generator_hidden_layers
        self.anomaly_generator_activation_function = anomaly_generator_activation_function
        self.anomaly_generator_dropout_fraction = anomaly_generator_dropout_fraction
        self.anomaly_generator_loss_function = anomaly_generator_loss_function
        self.anomaly_generator = None
        self.alarm_hidden_layers = alarm_hidden_layers
        self.alarm_activation_function = alarm_activation_function
        self.alarm_dropout_fraction = alarm_dropout_fraction
        self.alarm_loss_function = alarm_loss_function
        self._anomaly_generator_optimizer = _anomaly_generator_optimizer
        self._alarm_optimizer = _alarm_optimizer
        self.alarm = None
        self._anomaly_generator_epoch_mean_loss = tf.metrics.Mean()
        self._detector_epoch_mean_loss = tf.metrics.Mean()
        super().__init__(autoencoder_latent_dimension, encoder_hidden_layers, decoder_hidden_layers,
                         discrim_hidden_layers, autoencoder_activation, discrim_activation)

    def _build_anomaly_generator(self) -> None:
        generator_input = Input(shape=(self.anomaly_generator_input_size,))
        x = Dense(units=self.anomaly_generator_hidden_layers[0],
                  activation=self.anomaly_generator_activation_function)(generator_input)
        x = Dropout(self.anomaly_generator_dropout_fraction)(x)
        remaining_layers = self.anomaly_generator_hidden_layers[1:]
        for units in remaining_layers:
            x = Dense(units=units, activation=self.anomaly_generator_activation_function)(x)
            x = Dropout(self.anomaly_generator_dropout_fraction)(x)
        latent_code = Dense(self.autoencoder_latent_dimension,
                            activation=self.anomaly_generator_activation_function)(x) # TODO should maybe be linear
        anomaly_generator = Model(inputs=generator_input, outputs=latent_code)
        self.anomaly_generator = anomaly_generator

    def _build_alarm(self) -> None:
        alarm_input_size = np.sum(self.decoder_hidden_layers)
        alarm_input = Input(shape=(alarm_input_size,))
        x = Dense(units=self.alarm_hidden_layers[0], activation=self.alarm_activation_function)(alarm_input)
        x = Dropout(self.alarm_dropout_fraction)(x)
        remaining_layers = self.alarm_hidden_layers[1:]
        for units in remaining_layers:
            x = Dense(units=units, activation=self.alarm_activation_function)(x)
            x = Dropout(self.alarm_dropout_fraction)(x)
        anomalous_or_not = Dense(1, activation="linear")(x)  # linear output due to wasserstein loss
        alarm = Model(inputs=alarm_input, outputs=anomalous_or_not)
        self.alarm = alarm

    @staticmethod
    def _extract_activations(target_network: tf.keras.Model,
                             target_name: str) -> tf.keras.layers.Layer:
        """
        Get the activation layers of the defined model

        :param target_network: model to take the activation layers from
        :param target_name: give the target a name
        :return: a flattened layer with all hidden activations
        """

        all_activations = []
        # get all dense layers, but the last which is the output layer (we only want the hidden layers)
        hidden_dense_layers = [layer for layer in target_network.layers if isinstance(layer, tf.keras.layers.Dense)][
                              :-1]
        for layer_number, current_layer in enumerate(hidden_dense_layers):
            all_activations.append(tf.keras.layers.Flatten(name=f"{target_name}_{layer_number}")(current_layer.output))
        all_activations = tf.keras.layers.Concatenate(name=target_name)(all_activations)
        activation_model = tf.keras.models.Model(inputs=target_network.inputs, outputs=all_activations)  # testing this only
        return activation_model  # all_activations

    @staticmethod
    def _anomaly_generator_loss(alarm_output, discriminator_output):
        return tf.reduce_mean(alarm_output) - tf.reduce_mean(discriminator_output)

    @staticmethod
    def _anomaly_detector_loss(alarm_output_from_x,
                               alarm_output_from_generated_anomaly,
                               alarm_output_from_trivial_anomaly):
        loss_fun = tf.keras.losses.BinaryCrossentropy(from_logits=True,
                                                      reduction=tf.keras.losses.Reduction.NONE)
        loss = loss_fun(tf.zeros_like(alarm_output_from_x), alarm_output_from_x)
        loss += loss_fun(tf.ones_like(alarm_output_from_generated_anomaly), alarm_output_from_generated_anomaly)
        loss += loss_fun(tf.ones_like(alarm_output_from_trivial_anomaly), alarm_output_from_trivial_anomaly)
        loss = tf.reduce_mean(loss)
        return loss

    def _compute_DAD_losses_and_update_weights(self, batch_x):
        # TODO consider building these graphs outside the training loop
        # first define models/graphs to run data through:
        decoder_input = self.decoder.input
        decoder_activations = self._extract_activations(self.decoder, "decoder_activations")

        # (for generator) build hidden code ---> decoder activations ---> alarm output model: f_alarm(h)
        alarm_output = self.alarm(decoder_activations.output)
        decoder_input_to_alarm_model = tf.keras.models.Model(inputs=decoder_input, outputs=alarm_output)

        # (for generator) build hidden code ---> discriminator output model: f_disc(h)
        discriminator_output = self.discriminator(decoder_input)
        decoder_input_to_discrim_model = tf.keras.models.Model(inputs=decoder_input, outputs=discriminator_output)

        # (for generator) build x ---> hidden code ---> decoder activations model
        encoder_input = self.encoder.inputs
        hidden_code = self.encoder(encoder_input)
        decoder_hidden_activations = decoder_activations(hidden_code)
        encoder_input_to_decoder_activations_model = tf.keras.models.Model(inputs=encoder_input,
                                                                           outputs=decoder_hidden_activations)

        # (for detector) build x ---> hidden code ---> decoder activations ---> alarm output model: f_AD(x)
        alarm_output_from_encoder = self.alarm(encoder_input_to_decoder_activations_model.output)
        encoder_input_to_alarm_model = tf.keras.models.Model(inputs=encoder_input, outputs=alarm_output_from_encoder)

        # -------------------------------------------------------------------------------------------------------------
        # Anomaly generator network
        noise_for_generator = tf.random.normal([batch_x.shape[0], self.anomaly_generator_input_size],
                                               mean=0, stddev=1)  # n_gen
        with tf.GradientTape() as tape:
            generated_latent_anomaly = self.anomaly_generator(noise_for_generator, training=True)  # h^tilde
            alarm_output_from_generated_anomaly = decoder_input_to_alarm_model(generated_latent_anomaly,
                                                                               training=False)  # AG_1
            discriminator_output = decoder_input_to_discrim_model(generated_latent_anomaly, training=False)  # AG_2
            anomaly_generator_loss = self._anomaly_generator_loss(alarm_output_from_generated_anomaly,
                                                                  discriminator_output)

        gradients = tape.gradient(anomaly_generator_loss, self.anomaly_generator.trainable_variables)
        self._anomaly_generator_optimizer.apply_gradients(zip(gradients, self.anomaly_generator.trainable_variables))

        # -------------------------------------------------------------------------------------------------------------
        # Anomaly detector network
        noise_for_detector = tf.random.normal([batch_x.shape[0], self.number_of_features],
                                              mean=0.5, stddev=1)  # n_gen
        with tf.GradientTape() as tape:
            alarm_output_from_x = encoder_input_to_alarm_model(batch_x, training=True)  # AG_1
            alarm_output_from_generated_anomaly = decoder_input_to_alarm_model(generated_latent_anomaly,
                                                                               training=True)  # AG_1
            alarm_output_from_trivial_anomaly = encoder_input_to_alarm_model(noise_for_detector,
                                                                             training=True)
            anomaly_detector_loss = self._anomaly_detector_loss(alarm_output_from_x,
                                                                alarm_output_from_generated_anomaly,
                                                                alarm_output_from_trivial_anomaly)

        gradients = tape.gradient(anomaly_detector_loss, self.alarm.trainable_variables)
        self._anomaly_generator_optimizer.apply_gradients(zip(gradients, self.alarm.trainable_variables))

        return anomaly_generator_loss, anomaly_detector_loss

    def fit(self,
            df_features_scaled: DataFrame,
            batch_size: int = 128,
            epochs: int = 10,
            dropout_fraction: float = 0.2,
            base_learning_rate: float = 0.00025,
            max_learning_rate: float = 0.0025,
            verbose: int = 2):
        print("Starting AAE fit!")
        super().fit(df_features_scaled, batch_size, epochs, dropout_fraction, base_learning_rate, max_learning_rate,
                    verbose)
        print("Starting alarm and anomaly generator fit!")
        self._build_anomaly_generator()
        self._build_alarm()

        step_size = 2 * np.ceil(self.number_of_train_data_rows / self.batch_size)
        global_step = 0
        train_dataset_in_batches = self._prepare_dataset_in_batches(df_features_scaled)
        # Training loop
        for epoch in range(self.epochs):
            start = time.time()

            # Learning rate schedule
            if epoch in [60, 100, 300]:
                base_learning_rate = base_learning_rate / 2
                max_learning_rate = max_learning_rate / 2
                step_size = step_size / 2
                print('Learning rate decreased!')

            for batch_number, batch in enumerate(train_dataset_in_batches):
                if verbose > 2:
                    number_of_batches = int(np.floor(self.number_of_train_data_rows / self.batch_size))
                    print(f"Epoch {epoch + 1}, batch {batch_number + 1}/{number_of_batches + 1}")
                new_learning_rate = self._get_updated_learning_rate(base_learning_rate,
                                                                    global_step,
                                                                    max_learning_rate,
                                                                    step_size)
                self._anomaly_generator_optimizer.lr = new_learning_rate
                self._alarm_optimizer.lr = new_learning_rate
                losses = self._compute_DAD_losses_and_update_weights(batch)
                self._anomaly_generator_epoch_mean_loss(losses[0])
                self._detector_epoch_mean_loss(losses[1])

            epoch_time = time.time() - start
            if verbose > 1:
                print(
                    f'Epoch {epoch + 1}: Time: {epoch_time:.2f}s, ETA: {epoch_time * (self.epochs - epoch):.0f}s, '
                    f'detector loss: {self._detector_epoch_mean_loss.result():.4f}, '
                    f'generator loss: {self._anomaly_generator_epoch_mean_loss.result():.4f}'
                )
        if verbose > 0:
            print("\n... Fitting procedure complete - model fitted succesfully!")


if __name__ == "__main__":
    is_debugging = True
    # is_debugging = False
    pd.set_option('display.max_columns', None)
    filename = get_pump_data_path() / f"data_cwp_pump_10_{get_dataset_purpose_as_str(is_debugging)}.csv"
    df_features = get_local_data(filename)
    df_features = df_features.dropna()
    print(df_features.shape)

    scaler = MinMaxScaler()
    df_features_scaled = scaler.fit_transform(df_features)

    dad = DAD()
    dad.fit(df_features_scaled[:10000, ], epochs=10, batch_size=128, verbose=2)

    print("Finished")
