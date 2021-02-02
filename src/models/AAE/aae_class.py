import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from src.utilities.utility_functions import *
from src.data.data_utils import *
from src.config.definitions import *
import time

MODEL_NUMBER_TO_NAME_MAPPING = {"0": "encoder", "1": "decoder", "2": "discriminator"}


class AELossHandler:
    def __init__(self):
        self.autoencoder_epoch_mean_loss = tf.metrics.Mean()
        self.discriminator_epoch_mean_loss = tf.metrics.Mean()
        self.discriminator_epoch_mean_accuracy = tf.metrics.Mean()
        self.generator_epoch_mean_loss = tf.metrics.Mean()

    def update_losses(self,
                      autoencoder_loss,
                      discriminator_loss,
                      discriminator_accuracy,
                      generator_loss):
        self.autoencoder_epoch_mean_loss(autoencoder_loss)
        self.discriminator_epoch_mean_loss(discriminator_loss)
        self.discriminator_epoch_mean_accuracy(discriminator_accuracy)
        self.generator_epoch_mean_loss(generator_loss)

    def get_losses(self):
        """
        Get current mean losses from all the models
        :return: last recorded mean loss for all the models,
        :rtype: tuple
        """
        return (self.autoencoder_epoch_mean_loss.result(),
                self.discriminator_epoch_mean_loss.result(),
                self.discriminator_epoch_mean_accuracy.result(),
                self.generator_epoch_mean_loss.result())


# TODO add validation feature to training
# TODO add early stopping feature
class AdversarialAutoencoder:
    def __init__(self,
                 latent_layer_dimension: int = 2,
                 encoder_hidden_layers: List = None,
                 decoder_hidden_layers: List = None,
                 discrim_hidden_layers: List = None,
                 autoencoder_activation_function: tf.keras.layers = "elu",
                 discrim_activation_function: tf.keras.layers = "elu",
                 ):
        self.latent_layer_dimension = latent_layer_dimension
        self.encoder_hidden_layers = encoder_hidden_layers if encoder_hidden_layers is not None else [10, 8, 6, 4]
        self.decoder_hidden_layers = decoder_hidden_layers if decoder_hidden_layers is not None else [4, 6, 8, 10]
        self.discrim_hidden_layers = discrim_hidden_layers if discrim_hidden_layers is not None else [10, 8, 6, 4]
        self.autoencoder_activation_function = autoencoder_activation_function
        self.discrim_activation_function = discrim_activation_function
        self.discriminator = None
        self.encoder = None
        self.decoder = None
        self.epochs = None
        self.batch_size = None
        self.number_of_features = None
        self.number_of_train_data_rows = None
        self._autoencoder_optimizer = None
        self._discriminator_optimizer = None
        self._generator_optimizer = None

    def _build_encoder_model(self, dropout_fraction: float = 0.2):
        hidden_layers = self.encoder_hidden_layers
        inputs = tf.keras.Input(shape=(self.number_of_features,))
        x = tf.keras.layers.Dense(units=hidden_layers[0], activation=self.autoencoder_activation_function)(inputs)
        # x = self.autoencoder_activation_function()(x)
        x = tf.keras.layers.Dropout(dropout_fraction)(x)
        remaining_layers = hidden_layers[1:]
        for units in remaining_layers:
            x = tf.keras.layers.Dense(units=units, activation=self.autoencoder_activation_function)(x)
            # x = self.autoencoder_activation_function()(x)
            x = tf.keras.layers.Dropout(dropout_fraction)(x)
        latent_space = tf.keras.layers.Dense(self.latent_layer_dimension,
                                             activation=self.autoencoder_activation_function)(x)
        model = tf.keras.Model(inputs=inputs, outputs=latent_space)
        self.encoder = model

    def _build_decoder_model(self, dropout_fraction: float = 0.2):
        hidden_layers = self.decoder_hidden_layers
        latent_space = tf.keras.Input(shape=(self.latent_layer_dimension,))
        x = tf.keras.layers.Dense(units=hidden_layers[0],
                                  activation=self.autoencoder_activation_function)(latent_space)
        # x = self.autoencoder_activation_function()(x)
        x = tf.keras.layers.Dropout(dropout_fraction)(x)
        remaining_layers = hidden_layers[1:]
        for units in remaining_layers:
            x = tf.keras.layers.Dense(units=units, activation=self.autoencoder_activation_function)(x)
            # x = self.autoencoder_activation_function()(x)
            x = tf.keras.layers.Dropout(dropout_fraction)(x)
        reconstruction = tf.keras.layers.Dense(self.number_of_features, activation="linear")(x)
        model = tf.keras.Model(inputs=latent_space, outputs=reconstruction)
        self.decoder = model

    def _build_discriminator_model(self, dropout_fraction: float = 0.2):
        hidden_layers = self.discrim_hidden_layers
        latent_space = tf.keras.Input(shape=(self.latent_layer_dimension,))
        x = tf.keras.layers.Dense(units=hidden_layers[0],
                                  activation=self.discrim_activation_function)(latent_space)
        # x = self.discrim_activation_function()(x)
        x = tf.keras.layers.Dropout(dropout_fraction)(x)
        remaining_layers = hidden_layers[1:]
        for units in remaining_layers:
            x = tf.keras.layers.Dense(units=units, activation=self.discrim_activation_function)(x)
            # x = self.discrim_activation_function()(x)
            x = tf.keras.layers.Dropout(dropout_fraction)(x)
        prediction_real_or_fake = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        model = tf.keras.Model(inputs=latent_space, outputs=prediction_real_or_fake)
        self.discriminator = model

    @staticmethod
    def _reconstruction_loss(inputs, reconstruction, loss_weight: float = 1):
        mse = tf.keras.losses.MeanSquaredError()
        return loss_weight * mse(inputs, reconstruction)

    @staticmethod
    def _discriminator_loss(real_output, fake_output, loss_weight: float = 1):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        loss_real = cross_entropy(tf.ones_like(real_output), real_output)
        loss_fake = cross_entropy(tf.zeros_like(fake_output), fake_output)
        return loss_weight * (loss_fake + loss_real)

    @staticmethod
    def _generator_loss(fake_output, loss_weight: float = 1):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True) # TODO should this be Binary XE?
        return loss_weight * cross_entropy(tf.ones_like(fake_output), fake_output)

    def _prepare_dataset_in_batches(self, df_features: DataFrame):
        train_dataset = tf.data.Dataset.from_tensor_slices(df_features)
        train_dataset = train_dataset.shuffle(buffer_size=self.number_of_train_data_rows)
        train_dataset = train_dataset.batch(self.batch_size)
        return train_dataset

    @staticmethod
    def _get_updated_learning_rate(base_learning_rate, global_step, max_learning_rate, step_size):
        global_step = global_step + 1
        cycle = np.floor(1 + global_step / (2 * step_size))
        x_lr = np.abs(global_step / step_size - 2 * cycle + 1)
        new_learning_rate = base_learning_rate + (max_learning_rate - base_learning_rate) * max(0, 1 - x_lr)
        return new_learning_rate

    def _set_learning_rates(self, learning_rate):
        self._autoencoder_optimizer.lr = learning_rate
        self._discriminator_optimizer.lr = learning_rate
        self._generator_optimizer.lr = learning_rate

    def _compute_losses_and_update_weights(self, batch_x):
        def _update_model_weights(optimizer, gradients, trainable_variables):
            optimizer.apply_gradients(zip(gradients, trainable_variables))

        def _concat_tf_variables(tf_array_1, tf_array_2):
            return tf.concat([tf_array_1, tf_array_2], axis=0)

        # -------------------------------------------------------------------------------------------------------------
        # Autoencoder
        with tf.GradientTape() as autoencoder_tape:
            encoder_output = self.encoder(batch_x, training=True)
            decoder_output = self.decoder(encoder_output, training=True)
            reconstruction_loss = self._reconstruction_loss(batch_x, decoder_output, loss_weight=1)

        autoencoder_trainable_variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        autoencoder_gradients = autoencoder_tape.gradient(reconstruction_loss, autoencoder_trainable_variables)
        _update_model_weights(self._autoencoder_optimizer, autoencoder_gradients, autoencoder_trainable_variables)

        # -------------------------------------------------------------------------------------------------------------
        # Discriminator
        with tf.GradientTape() as discriminator_tape:
            # TODO make this more generic, so arbitrary distributions can be given
            real_distribution = tf.random.normal([batch_x.shape[0], self.latent_layer_dimension],
                                                 mean=0, stddev=0.01)
            encoder_output = self.encoder(batch_x, training=True)

            discriminator_real_sample_output = self.discriminator(real_distribution, training=True)
            discriminator_fake_sample_output = self.discriminator(encoder_output, training=True)

            discriminator_loss = self._discriminator_loss(discriminator_real_sample_output,
                                                          discriminator_fake_sample_output, loss_weight=1)
            accuracy = tf.keras.metrics.BinaryAccuracy()
            discriminator_true_labels = _concat_tf_variables(tf.ones_like(discriminator_real_sample_output),
                                                             tf.zeros_like(discriminator_fake_sample_output))
            discriminator_predicted_labels = _concat_tf_variables(discriminator_real_sample_output,
                                                                  discriminator_fake_sample_output)
            discriminator_accuracy = accuracy(discriminator_true_labels, discriminator_predicted_labels)

        discriminator_gradients = discriminator_tape.gradient(discriminator_loss,
                                                              self.discriminator.trainable_variables)
        _update_model_weights(self._discriminator_optimizer, discriminator_gradients,
                              self.discriminator.trainable_variables)

        # -------------------------------------------------------------------------------------------------------------
        # Generator (Encoder)
        with tf.GradientTape() as generator_tape:
            encoder_output = self.encoder(batch_x, training=True)
            discriminator_fake_sample_output = self.discriminator(encoder_output, training=True)

            generator_loss = self._generator_loss(discriminator_fake_sample_output, loss_weight=1)

        generator_gradients = generator_tape.gradient(generator_loss, self.encoder.trainable_variables)
        _update_model_weights(self._generator_optimizer, generator_gradients, self.encoder.trainable_variables)

        return reconstruction_loss, discriminator_loss, discriminator_accuracy, generator_loss

    def fit(self,
            df_features_scaled: DataFrame,
            batch_size: int = 128,
            epochs: int = 10,
            dropout_fraction: float = 0.2,
            base_learning_rate: float = 0.00025,
            max_learning_rate: float = 0.0025,
            verbose: int = 2):

        self.number_of_train_data_rows = df_features_scaled.shape[0]
        self.number_of_features = df_features_scaled.shape[1]
        self.batch_size = batch_size
        self.epochs = epochs
        self._build_encoder_model(dropout_fraction=dropout_fraction)
        self._build_decoder_model(dropout_fraction=dropout_fraction)
        self._build_discriminator_model(dropout_fraction=dropout_fraction)
        self._autoencoder_optimizer = tf.keras.optimizers.Adam(learning_rate=base_learning_rate)
        self._discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=base_learning_rate)
        self._generator_optimizer = tf.keras.optimizers.Adam(learning_rate=base_learning_rate)

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

            loss_handler = AELossHandler()
            for batch_number, batch in enumerate(train_dataset_in_batches):
                if verbose > 2:
                    number_of_batches = int(np.floor(self.number_of_train_data_rows / batch_size))
                    print(f"Epoch {epoch + 1}, batch {batch_number + 1}/{number_of_batches + 1}")
                new_learning_rate = self._get_updated_learning_rate(base_learning_rate,
                                                                    global_step,
                                                                    max_learning_rate,
                                                                    step_size)
                self._set_learning_rates(new_learning_rate)
                losses = self._compute_losses_and_update_weights(batch)

                loss_handler.update_losses(*losses)

            epoch_time = time.time() - start
            if verbose > 1:
                print(
                    'Epoch {}: Time: {:.2f}s, ETA: {:.0f}s, AE loss: {:.4f}, discriminator loss: {:.4f}, '
                    'discriminator accuracy: {:.4f}, generator loss: {:.4f}'.format(epoch+1, epoch_time,
                                                                                    epoch_time * (self.epochs - epoch),
                                                                                    *loss_handler.get_losses()
                                                                                    )
                )
        if verbose > 0:
            print("\n... Fitting procedure complete - model fitted succesfully!")

    def predict(self, df_features_scaled):
        latent_space = self.encoder.predict(df_features_scaled)
        return self.decoder.predict(latent_space)

    def predict_latent_space(self, df_features_scaled):
        return self.encoder.predict(df_features_scaled)

    def plot_latent_space(self, df_features_scaled, coloring_column: DataFrame = None):
        df_latent_space = pd.DataFrame(self.predict_latent_space(df_features_scaled))
        title_latent = "AAE latent space"
        pca_plot = self.latent_layer_dimension > 2
        if pca_plot:
            pca = PCA(n_components=2)
            df_latent_space = pd.DataFrame(pca.fit_transform(df_latent_space))
            var_expl = 100 * pca.explained_variance_ratio_.sum()
            title_latent = title_latent + f"\nPCA transformed (variance explained:{var_expl:4.0f}%)"
        if coloring_column is None:
            color = pd.DataFrame(df_features_scaled).iloc[:, 0]
            colorbar_label = "First (scaled) feature column"
        else:
            color = coloring_column.iloc[:, 0]
            colorbar_label = coloring_column.columns.values[0]

        plt.scatter(df_latent_space.iloc[:, 0], df_latent_space.iloc[:, 1],
                    c=color, cmap='jet', s=10)
        plt.xlabel("PC1" if pca_plot else "z0")
        plt.ylabel("PC2" if pca_plot else "z1")
        clb = plt.colorbar()
        clb.set_label(colorbar_label, rotation=0, labelpad=-30, y=1.05)
        plt.title(title_latent)
        plt.show()

    def save(self, filename: str = None):
        model_list = [self.encoder, self.decoder, self.discriminator]
        for model_number, model in enumerate(model_list):
            model_name = MODEL_NUMBER_TO_NAME_MAPPING[str(model_number)]
            if filename is None:
                filename = get_model_archive_path() / model_name
            else:
                filename = get_model_archive_path() / f"{filename}_{model_name}"
            tf.keras.models.save_model(model, filename)
        print(f"Model saved succesfully in {get_model_archive_path()}")

    def load(self, filename: str = None):
        model_list = [self.encoder, self.decoder, self.discriminator]
        loaded_models = {"encoder": None, "decoder": None, "discriminator": None}
        for model_number, loaded_model in enumerate(model_list):
            model = MODEL_NUMBER_TO_NAME_MAPPING[str(model_number)]
            if filename is None:
                filename = get_model_archive_path() / model
            else:
                filename = get_model_archive_path() / f"{filename}_{model}"
            loaded_models[model] = tf.keras.models.load_model(filename, compile=False)
        self.encoder = loaded_models["encoder"]
        self.decoder = loaded_models["decoder"]
        self.discriminator = loaded_models["discriminator"]
        print(f"Model loaded succesfully from {get_model_archive_path()}")



if __name__ == "__main__":
    is_debugging = True
    # is_debugging = False
    pd.set_option('display.max_columns', None)
    filename = get_pump_data_path() / f"data_cwp_pump_10_{get_dataset_purpose_as_str(is_debugging)}.csv"
    df_features = get_local_data(filename)
    df_features = df_features.dropna()
    print(df_features.shape)

    aae = AdversarialAutoencoder(latent_layer_dimension=2,
                                 encoder_hidden_layers=[6, 4],
                                 decoder_hidden_layers=[4, 6],
                                 discrim_hidden_layers=[6, 4, 2])
    scaler = MinMaxScaler()
    df_features_scaled = scaler.fit_transform(df_features)
    aae.fit(df_features_scaled, dropout_fraction=0.3, epochs=1, batch_size=1024, verbose=3)
    aae.save("ssv_cwp_pump10_aae")
    aae = AdversarialAutoencoder()
    aae.load("ssv_cwp_pump10_aae")
    foo = aae.predict_latent_space(df_features_scaled)
    aae.plot_latent_space(df_features_scaled, df_features[["flow"]])
