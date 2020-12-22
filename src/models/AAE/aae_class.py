import tensorflow as tf
import numpy as np
from pandas import DataFrame
from dataclasses import dataclass, field
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from src.models.AAE.aae_utils import *
from src.models.AAE.definitions import *
import time


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
class AdversarialAutoencoder:
    def __init__(self, first_hidden_layer_dimension: int, latent_layer_dimension: int):
        self.first_hidden_layer_dimension = first_hidden_layer_dimension
        self.latent_layer_dimension = latent_layer_dimension
        # TODO add options for discriminator dimensionality, too
        self.discriminator = None
        self.encoder = None
        self.decoder = None
        self.epochs = None
        self.batch_size = None
        self.number_of_features = None
        self.number_of_train_data_rows = None
        self.autoencoder_optimizer = None
        self.discriminator_optimizer = None
        self.generator_optimizer = None
        # TODO architecture choices should be taken by __init__ method

    def _prepare_dataset_in_batches(self, df_features: DataFrame):
        train_dataset = tf.data.Dataset.from_tensor_slices(df_features)
        train_dataset = train_dataset.shuffle(buffer_size=self.number_of_train_data_rows)
        train_dataset = train_dataset.batch(self.batch_size)
        return train_dataset

    def _get_updated_learning_rate(self, base_learning_rate, global_step, max_learning_rate, step_size):
        global_step = global_step + 1
        cycle = np.floor(1 + global_step / (2 * step_size))
        x_lr = np.abs(global_step / step_size - 2 * cycle + 1)
        new_learning_rate = base_learning_rate + (max_learning_rate - base_learning_rate) * max(0, 1 - x_lr)
        return new_learning_rate

    def _set_learning_rates(self, learning_rate):
        self.autoencoder_optimizer.lr = learning_rate
        self.discriminator_optimizer.lr = learning_rate
        self.generator_optimizer.lr = learning_rate

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
        _update_model_weights(self.autoencoder_optimizer, autoencoder_gradients, autoencoder_trainable_variables)

        # -------------------------------------------------------------------------------------------------------------
        # Discriminator
        with tf.GradientTape() as discriminator_tape:
            # TODO make this more generic, so arbitrary distributions can be given
            real_distribution = tf.random.normal([batch_x.shape[0], self.latent_layer_dimension], mean=0.0, stddev=1.0)
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
        _update_model_weights(self.discriminator_optimizer, discriminator_gradients,
                              self.discriminator.trainable_variables)

        # -------------------------------------------------------------------------------------------------------------
        # Generator (Encoder)
        with tf.GradientTape() as generator_tape:
            encoder_output = self.encoder(batch_x, training=True)
            discriminator_fake_sample_output = self.discriminator(encoder_output, training=True)

            generator_loss = self._generator_loss(discriminator_fake_sample_output, loss_weight=1)

        generator_gradients = generator_tape.gradient(generator_loss, self.encoder.trainable_variables)
        _update_model_weights(self.generator_optimizer, generator_gradients, self.encoder.trainable_variables)

        return reconstruction_loss, discriminator_loss, discriminator_accuracy, generator_loss

    def fit(self, df_features_scaled: DataFrame,
            batch_size: int = 128,
            epochs: int = 10,
            base_learning_rate: float = 0.00025,
            max_learning_rate: float = 0.0025):

        self.number_of_train_data_rows = df_features_scaled.shape[0]
        self.number_of_features = df_features_scaled.shape[1]
        self.batch_size = batch_size
        self.epochs = epochs
        self.encoder = self._make_encoder_model()
        self.decoder = self._make_decoder_model()
        self.discriminator = self._make_discriminator_model()
        self.autoencoder_optimizer = tf.keras.optimizers.Adam(learning_rate=base_learning_rate)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=base_learning_rate)
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=base_learning_rate)

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
                new_learning_rate = self._get_updated_learning_rate(base_learning_rate,
                                                                    global_step,
                                                                    max_learning_rate,
                                                                    step_size)
                self._set_learning_rates(new_learning_rate)
                losses = self._compute_losses_and_update_weights(batch)

                loss_handler.update_losses(*losses)

            epoch_time = time.time() - start
            print(
                'Epoch {:3d}: Time: {:.2f}s, ETA: {:.0f}s, AE loss: {:.4f}, Discriminator loss: {:.4f}, '
                'discriminator accuracy: {:.4f}, generator loss: {:.4f}'.format(epoch, epoch_time,
                                                                                epoch_time * (self.epochs - epoch),
                                                                                *loss_handler.get_losses()
                                                                                )
            )

    def predict(self, df_features_scaled):
        latent_space = self.encoder.predict(df_features_scaled)
        return self.decoder.predict(latent_space)

    def predict_latent_space(self, df_features_scaled):
        return self.encoder.predict(df_features_scaled)

    def _make_encoder_model(self):
        inputs = tf.keras.Input(shape=(self.number_of_features,))
        x = tf.keras.layers.Dense(self.first_hidden_layer_dimension)(inputs)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(self.first_hidden_layer_dimension - 2)(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        encoded = tf.keras.layers.Dense(self.latent_layer_dimension)(x)
        model = tf.keras.Model(inputs=inputs, outputs=encoded)
        return model

    def _make_decoder_model(self):
        encoded = tf.keras.Input(shape=(self.latent_layer_dimension,))
        x = tf.keras.layers.Dense(self.first_hidden_layer_dimension - 2)(encoded)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(self.first_hidden_layer_dimension)(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        reconstruction = tf.keras.layers.Dense(self.number_of_features, activation='sigmoid')(x)
        model = tf.keras.Model(inputs=encoded, outputs=reconstruction)
        return model

    def _make_discriminator_model(self):
        encoded = tf.keras.Input(shape=(self.latent_layer_dimension,))
        x = tf.keras.layers.Dense(self.first_hidden_layer_dimension)(encoded)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(self.first_hidden_layer_dimension - 2)(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        prediction_real_or_fake = tf.keras.layers.Dense(1)(x)
        model = tf.keras.Model(inputs=encoded, outputs=prediction_real_or_fake)
        return model

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
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        return loss_weight * cross_entropy(tf.ones_like(fake_output), fake_output)


if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    df_features = get_local_pump_data(small_dataset=True, station="SSV", component="CWP", pump_number="10")

    aae = AdversarialAutoencoder(first_hidden_layer_dimension=12, latent_layer_dimension=2)
    scaler = MinMaxScaler()
    df_features_scaled = scaler.fit_transform(df_features)
    aae.fit(df_features_scaled)
