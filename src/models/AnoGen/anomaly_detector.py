from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import keras
import pickle
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense
from keras.models import Model, Sequential
from numpy.random import normal
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, average_precision_score, roc_auc_score, roc_curve
from pdm.models.clustering.dbscan import DBSCANClustering
from dataclasses import dataclass


from src.models.AnoGen.utility_functions import fit_VAE


class AnomalyDetectorAutoencoder:
    def __init__(self,
                 first_hidden_layer_dimension,
                 latent_space_dimension
                 ):
        self.latent_space_dimension = latent_space_dimension
        self.first_hidden_layer_dimension = first_hidden_layer_dimension
        self.keras_model = None
        self.anomaly_threshold = None
        self.mse_train_set_actual_vs_predicted = None
        self.mse_val_set_actual_vs_predicted = None
        self.epochs = None
        self.early_stopping = None
        self.loss_function = None
        self.batch_size = None
        self.activation_function = None
        self.scaler = None

    def fit(self,
            df_x_train,
            df_x_val,
            epochs=200,
            early_stopping=True,
            loss_function="mse",
            batch_size=128,
            activation_function="tanh",
            plot_learning_curve=False,
            plot_latent_space=False):
        """
        Trains an ordinary autoencoder for anomaly detection
        """

        self.scaler = MinMaxScaler()
        x_train_scaled = self.scaler.fit_transform(df_x_train)
        x_val_scaled = self.scaler.transform(df_x_val)

        callbacks_to_keras_model = []
        if early_stopping:
            callbacks_to_keras_model.append(EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True))

        feature_space_dimension = x_train_scaled.shape[1]

        x_input = Input(shape=(feature_space_dimension,))
        hidden_layer = Dense(self.first_hidden_layer_dimension, activation=activation_function)(x_input)
        hidden_layer = Dense(self.first_hidden_layer_dimension - 2, activation=activation_function)(hidden_layer)
        hidden_layer = Dense(self.first_hidden_layer_dimension - 4, activation=activation_function)(hidden_layer)
        latent_layer = Dense(self.latent_space_dimension, activation=activation_function)(hidden_layer)
        hidden_layer = Dense(self.first_hidden_layer_dimension - 4, activation=activation_function)(latent_layer)
        hidden_layer = Dense(self.first_hidden_layer_dimension - 2, activation=activation_function)(hidden_layer)
        hidden_layer = Dense(self.first_hidden_layer_dimension, activation=activation_function)(hidden_layer)
        x_input_reconstructed = Dense(feature_space_dimension, activation=activation_function)(hidden_layer)

        encoder_model = Model(inputs=x_input, outputs=latent_layer)

        autoencoder_model = Model(inputs=x_input, outputs=x_input_reconstructed, name='ae')
        autoencoder_model.compile(optimizer='adam', loss=loss_function)

        hist = autoencoder_model.fit(
            x_train_scaled,
            x_train_scaled,
            shuffle=False,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_val_scaled, x_val_scaled),
            verbose=2,
            callbacks=callbacks_to_keras_model
        )

        train_set_predictions = autoencoder_model.predict(x_train_scaled)
        val_set_predictions = autoencoder_model.predict(x_val_scaled)
        mse_train_set_actual_vs_predicted = np.mean(np.power(x_train_scaled - train_set_predictions, 2), axis=1)
        mse_val_set_actual_vs_predicted = np.mean(np.power(x_val_scaled - val_set_predictions, 2), axis=1)

        if plot_latent_space:
            z_train = encoder_model.predict(x_train_scaled)
            title_latent = "Autoencoder latent space (training data)"
            if self.latent_space_dimension > 2:
                reduction_method = "pca"
                if reduction_method.lower() == "pca":
                    pca = PCA(n_components=2).fit(z_train)
                    z_train = pca.transform(z_train)
                    var_expl = 100 * pca.explained_variance_ratio_.sum()
                    title_latent = title_latent + "\nPCA reduced (var explained: {0:4.0f})%".format(var_expl)
                else:
                    z_train = TSNE(n_components=2, learning_rate=75).fit_transform(z_train)
                    title_latent = title_latent + "\nVisualized using t-SNE"
            plt.scatter(z_train[:, 0], z_train[:, 1], s=10)
            plt.xlabel("z_0")
            plt.ylabel("z_1")
            plt.title(title_latent)
            plt.show()

        if plot_learning_curve:
            fig, ax = plt.subplots()
            hist_df = pd.DataFrame(hist.history)
            hist_df.plot(ax=ax)
            plt.suptitle("Autoencoder learning curve")
            ax.set_ylabel('Loss')
            ax.set_xlabel('# epochs')
            plt.show()

        self.keras_model = autoencoder_model
        self.mse_train_set_actual_vs_predicted = mse_train_set_actual_vs_predicted
        self.mse_val_set_actual_vs_predicted = mse_val_set_actual_vs_predicted
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.loss_function = loss_function
        self.batch_size = batch_size
        self.activation_function = activation_function
        return self

    def predict_new_and_return_mse(self, x_unscaled_to_predict):
        x_scaled_to_predict = self.scaler.transform(x_unscaled_to_predict)
        x_scaled_predicted = self.keras_model.predict(x_scaled_to_predict)
        mse_actual_vs_predicted = np.mean(np.power(np.array(x_scaled_to_predict) - x_scaled_predicted, 2), axis=1)
        df_mse_actual_vs_predicted = pd.DataFrame({"mse_actual_vs_observed": mse_actual_vs_predicted})
        return df_mse_actual_vs_predicted

    @staticmethod
    def compute_binary_anomalies_from_mse_and_threshold(mse_actual_vs_observed, anomaly_threshold):
        boolean_anomaly_labels = [data_point_mse > anomaly_threshold for data_point_mse in mse_actual_vs_observed]
        binary_anomaly_labels = np.array(boolean_anomaly_labels).astype(int)
        return binary_anomaly_labels

    @staticmethod
    def plot_predictions(df_anomaly_predictions):
        df_anomaly_predictions.plot(subplots=True, layout=(df_anomaly_predictions.shape[1], 1))
        plt.suptitle("Anomaly detector predictions")
        plt.show()

    def save(self, filename="AnomalyDetectorAE.pickle"):
        with open(filename, "wb") as file:
            pickle.dump(self, file)

    def load(self, filename="AnomalyDetectorAE.pickle"):
        with open(filename, "rb") as file:
            self.__dict__.update(pickle.load(file).__dict__)
