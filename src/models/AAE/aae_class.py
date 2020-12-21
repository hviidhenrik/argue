import tensorflow as tf
import numpy as np
from pandas import DataFrame
from dataclasses import dataclass, field
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from src.models.AAE.aae_utils import *
from src.models.AAE.definitions import *

@dataclass()
class AAEDataset:
    dataset: DataFrame
    batch_size: int
    rows: int = field(init=False)
    columns: int = field(init=False)

    def __post_init__(self):
        self.rows, self.columns = self.dataset.shape

    def make_batches(self):
        train_dataset = tf.data.Dataset.from_tensor_slices(self.dataset)
        train_dataset = train_dataset.shuffle(buffer_size=self.dataset.shape[0])
        train_dataset = train_dataset.batch(self.batch_size)
        return train_dataset


class AdversarialAutoencoder(AAEDataset):
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
        # TODO architecture choices should be taken by __init__ method

    def _prepare_dataset_in_batches(self, df_features: DataFrame):
        train_dataset = tf.data.Dataset.from_tensor_slices(df_features)
        train_dataset = train_dataset.shuffle(buffer_size=self.number_of_train_data_rows)
        train_dataset = train_dataset.batch(self.batch_size)
        return train_dataset

    def fit(self, df_features: DataFrame, batch_size: int = 128, epochs: int = 10):
        self.number_of_train_data_rows = df_features.shape[0]
        self.number_of_features = df_features.shape[1]
        self.batch_size = batch_size
        self.epochs = epochs
        self.encoder = self._make_encoder_model()
        self.decoder = self._make_decoder_model()
        self.discriminator = self._make_discriminator_model()

        # prepare data
        batch_dataset = self._prepare_dataset_in_batches(df_features)

        # fit AAE
        self._train_step()

    def predict(self):
        pass


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

    def _train_step(self, batch_x):
        # Define cyclic learning rate
        base_lr = 0.00025
        max_lr = 0.0025

        n_samples = x_train.shape[0]
        step_size = 2 * np.ceil(n_samples / batch_size)
        global_step = 0

        # -------------------------------------------------------------------------------------------------------------
        # Define optimizers
        ae_optimizer = tf.keras.optimizers.Adam(lr=base_lr)
        dc_optimizer = tf.keras.optimizers.Adam(lr=base_lr)
        gen_optimizer = tf.keras.optimizers.Adam(lr=base_lr)
        # -------------------------------------------------------------------------------------------------------------
        # Autoencoder
        with tf.GradientTape() as ae_tape:
            encoder_output = encoder(batch_x, training=True)
            decoder_output = decoder(encoder_output, training=True)

            # Autoencoder loss
            ae_loss = autoencoder_loss(batch_x, decoder_output, ae_loss_weight)

        ae_grads = ae_tape.gradient(ae_loss, encoder.trainable_variables + decoder.trainable_variables)
        ae_optimizer.apply_gradients(zip(ae_grads, encoder.trainable_variables + decoder.trainable_variables))

        # -------------------------------------------------------------------------------------------------------------
        # Discriminator
        with tf.GradientTape() as dc_tape:
            real_distribution = tf.random.normal([batch_x.shape[0], z_dim], mean=0.0, stddev=1.0)
            encoder_output = encoder(batch_x, training=True)

            dc_real = discriminator(real_distribution, training=True)
            dc_fake = discriminator(encoder_output, training=True)

            # Discriminator Loss
            dc_loss = discriminator_loss(dc_real, dc_fake, dc_loss_weight)

            # Discriminator Acc
            dc_acc = accuracy(tf.concat([tf.ones_like(dc_real), tf.zeros_like(dc_fake)], axis=0),
                              tf.concat([dc_real, dc_fake], axis=0))

        dc_grads = dc_tape.gradient(dc_loss, discriminator.trainable_variables)
        dc_optimizer.apply_gradients(zip(dc_grads, discriminator.trainable_variables))

        # -------------------------------------------------------------------------------------------------------------
        # Generator (Encoder)
        with tf.GradientTape() as gen_tape:
            encoder_output = encoder(batch_x, training=True)
            dc_fake = discriminator(encoder_output, training=True)

            # Generator loss
            gen_loss = generator_loss(dc_fake, gen_loss_weight)

        gen_grads = gen_tape.gradient(gen_loss, encoder.trainable_variables)
        gen_optimizer.apply_gradients(zip(gen_grads, encoder.trainable_variables))

        return ae_loss, dc_loss, dc_acc, gen_loss


if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    df_features = get_local_pump_data(small_dataset=True, station="SSV", component="CWP", pump_number="10")
    dataset = AAEDataset(df_features, 256).make_batches()

