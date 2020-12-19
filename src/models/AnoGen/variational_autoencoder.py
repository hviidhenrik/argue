import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Dense, Lambda, Add, Multiply, Layer, BatchNormalization, Dropout
from keras.models import Model, Sequential, load_model
from keras.callbacks import EarlyStopping, LambdaCallback
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import *
import numpy as np
from numpy.random import normal
from tensorflow.python.ops import math_ops


def nll(y_true, y_pred):
    """ Negative log likelihood (Bernoulli).

    TODO: implement Gaussian version too
    """
    # keras.losses.binary_crossentropy gives the mean
    # over the last axis. we require the sum
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)


class KLDivergenceLayer(Layer):
    """ Identity transform layer that adds KL divergence
    to the final model loss_function.
    """

    def __init__(self, warm_up=False, beta=1.0, *args, **kwargs):
        self.is_placeholder = True
        self.beta = beta
        self.warm_up = warm_up
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({"beta": self.beta})
        return config

    def call(self, inputs):
        mu, log_var = inputs
        kl_batch = - .5 * K.sum(1 + log_var - K.square(mu) - K.exp(log_var), axis=-1)
        if self.warm_up:  # if warm up is turned on
            # self.add_metric(self.beta, name="KL_weight")
            kl_batch = K.mean(kl_batch) * self.beta
        else:
            kl_batch = K.mean(kl_batch)
        self.add_metric(math_ops.reduce_mean(kl_batch), name="KL")
        self.add_loss(kl_batch, inputs=inputs)

        return inputs


def fit_VAE(x_train_scaled,
            x_val_scaled,
            intermediate_dim=9,
            latent_dim=2,
            batch_size=100,
            epochs=10,
            early_stopping=True,
            kl_warmup=10,
            latent_stddev=1.0,
            plot_history=False,
            activation="elu"):
    """
    Trains a variational autoencoder

    """

    callbacks = []
    if early_stopping:
        callbacks.append(EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True))

    kl_beta = False
    if kl_warmup:
        kl_beta = K.variable(1.0, name="kl_beta")
        kl_beta._trainable = False
        callbacks.append(LambdaCallback(
            on_epoch_begin=lambda epoch, logs: K.set_value(kl_beta, K.min([epoch / kl_warmup, 1]))
        )
        )

    # Specify hyperparameters
    original_dim = x_train_scaled.shape[1]

    # Encoder
    x = Input(shape=(original_dim,))
    h = Dropout(0.15)(x)
    h = Dense(intermediate_dim, activation=activation)(h)
    # h = BatchNormalization()(h)
    h = Dense(intermediate_dim - 2, activation=activation)(h)
    # h = BatchNormalization()(h)
    h = Dropout(0.15)(h)
    h = Dense(intermediate_dim - 4, activation=activation)(h)

    # bottleneck
    z_mu = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)

    z_mu, z_log_var = KLDivergenceLayer(beta=kl_beta, warm_up=kl_warmup > 0)([z_mu, z_log_var])

    # Reparametrization trick
    z_sigma = Lambda(lambda t: K.exp(.5 * t))(z_log_var)
    eps = Input(tensor=K.random_normal(shape=(K.shape(x)[0], latent_dim),
                                       mean=0,
                                       stddev=latent_stddev))
    z_eps = Multiply()([z_sigma, eps])
    z = Add()([z_mu, z_eps])

    # This defines the Encoder which takes noise and input, and outputs
    # the latent variable z
    encoder = Model(inputs=[x, eps], outputs=z)

    # Decoder is MLP specified as single Keras Sequential Layer
    decoder = Sequential([
        Dense(intermediate_dim - 4, input_dim=latent_dim, activation=activation),
        Dropout(0.15),
        # BatchNormalization(),
        Dense(intermediate_dim - 2, input_dim=latent_dim, activation=activation),
        # BatchNormalization(),
        Dense(intermediate_dim, input_dim=latent_dim, activation=activation),
        Dropout(0.15),
        Dense(original_dim, activation='tanh')
    ])

    x_pred = decoder(z)

    vae = Model(inputs=[x, eps], outputs=x_pred, name='vae')
    vae.compile(optimizer='adam', loss=nll)

    hist = vae.fit(
        x_train_scaled,
        x_train_scaled,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val_scaled, x_val_scaled),
        verbose=2,
        callbacks=callbacks
    )
    if plot_history:
        # Training loss_function plot
        fig, ax = plt.subplots()
        hist_df = pd.DataFrame(hist.history)
        hist_df.plot(ax=ax)
        plt.suptitle("Variational Autoencoder learning curve")
        ax.set_ylabel('NELBO')
        ax.set_xlabel('# epochs')
        plt.show()

    return encoder, decoder, vae


# fixme doesn't work right now
def load_VAE(filenames):
    """
    Doesnt work right now

    :param filenames:
    :type filenames:
    :return:
    :rtype:
    """
    vae = load_model("vae_model", custom_objects={'KLDivergenceLayer': KLDivergenceLayer,
                                                  'nll': nll})
    encoder = load_model("vae_encoder", custom_objects={'KLDivergenceLayer': KLDivergenceLayer,
                                                        'nll': nll})
    decoder = load_model("vae_decoder", custom_objects={'KLDivergenceLayer': KLDivergenceLayer,
                                                        'nll': nll})
    return vae, encoder, decoder
