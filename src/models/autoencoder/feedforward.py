from typing import List, Any, Union, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from pathlib import WindowsPath
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Nadam
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from src.models.autoencoder.base import AutoencoderMixin
from src.models.base.plot_mixin import PlotsMixin, save_or_show_plot


class FFNAutoencoder(AutoencoderMixin, PlotsMixin):
    def __init__(
        self,
        cols_to_include: List[str] = None,
        mse_quantile: float = 0.995,
        scaler: StandardScaler = None,
        epochs: int = 250,
        batch_size: int = 128,
        optimizer: Any = Nadam(learning_rate=0.001),
        plot: bool = False,
        verbose: bool = True,
        hidden_layers: List[int] = None,
        activation_functions: Union[str, List] = "tanh",
        dropout_fraction: float = 0,
        loss_function: str = "mse",
    ):
        """
        Instantiates a Feed-Forward-Network Autoencoder model, i.e. your good old ordinary dense autoencoder.

        :param cols_to_include: DOESNT DO THIS ATM: a list specifying the data columns used by the model
        :type cols_to_include: List[str], default=None
        :param mse_quantile the quantile from the test set to use as anomaly threshold
        :type mse_quantile float, default=0.995
        :param scaler: which scaler to use for scaling the data prior to modelling
        :type scaler: scaler object, default=StandardScaler
        :param epochs: number of full passes over the dataset
        :type epochs: int, default=250
        :param batch_size: number of samples considered for each weight update by a step in gradient descent
        :type batch_size: int, default=128
        :param optimizer: the optimizer to use for updating the weights during training, keras optimizer or string
        :type optimizer: Union[keras.optimizer, str], default=Nadam(learning_rate=0.001)
        :param plot: plot the training results or not
        :type plot: bool, default=True
        :param verbose: should the model fitting be informative or not
        :type verbose: bool, default=True
        :param hidden_layers a list of integer numbers of units in the layers between input and output
        :type hidden_layers List[int], default=[5, 2, 5]
        :param activation_functions a string specifying the activation function to be used.
               Should either be a single string, or a list of strings with Length |hidden_layers| + 1,
               e.g. activation_functions = ["tanh", "relu", "tanh", "linear"] with hidden_layers = [3,2,3].
               If a single string is given, e.g. "tanh", this will be used for all hidden layers and a linear
               activation will be used for the output layer.
               See Keras documentation for a list of activation functions:
               https://keras.io/api/layers/activations/
        :param dropout_fraction a fraction in [0,1) of the percentage of units to dropout in each layer
        :type dropout_fraction float, default=0
        :param loss_function the loss function to use - should be either "mse" or "nll" (negative log-likelihood)
        :type loss_function str, default="mse"
        """
        self.model = None
        self.cols_to_include = cols_to_include
        self.mse_quantile = mse_quantile
        self.anomaly_threshold = None
        self.scaler = scaler
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.plot = plot
        self.verbose = verbose
        self.hidden_layers = [5, 2, 5] if hidden_layers is None else hidden_layers
        self.activation_functions = activation_functions
        self.dropout_fraction = dropout_fraction
        self.loss_function = loss_function
        self.N_features = None
        self.latent_dimension = np.min(hidden_layers)
        self.encoder = None

    def fit(
        self,
        x_train: DataFrame,
        x_test: DataFrame,
        early_stopping: bool = False,
        early_stopping_patience: int = 10,
        plot_save_folder_path: Union[str, WindowsPath] = None,
        **kwargs,
    ):
        """
        Method that handles training of the autoencoder with associated hyperparameters and architecture of the model

        :param x_train: The training dataframe
        :type x_train: DataFrame
        :param x_test: The validation dataframe
        :type x_test: DataFrame
        :param early_stopping should the model fitting stop if validation loss improvement takes place?
        :type early_stopping bool, default=False
        :param early_stopping_patience the amount of epochs that no improvement should happen before stopping
        :type early_stopping_patience int, default=10
        :param plot_save_folder_path: a path to a folder in which to save all plots produced by the autoencoder. Leave
        as None, if no saving is desired.
        :type plot_save_folder_path Union[str, WindowsPath], default=None
        :param kwargs for now, can be a string specifying a reference col from the dataset for the
        latent space plot. Must be named "latent_reference_col".
        :return: the FFNAutoencoder instance itself
        :rtype: FFNAutoencoder
        """

        if self.verbose:
            print("\n===== Autoencoder =====")
            print("Training autoencoder model with {0} input features:".format(x_train.shape[1]))
            print(x_train.columns)

        self.N_features = x_train.shape[1]
        col_names = x_train.columns
        self.cols_to_include = list(col_names)
        col_names_pred = col_names + "_pred"
        date_index = x_test.index

        self._build_autoencoder()
        if self.verbose:
            self.model.summary()

        autoencoder = self.model
        callbacks = []
        if early_stopping:
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True,
                )
            )
        history = self.model.fit(
            x_train,
            x_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(x_test, x_test),
            verbose=self.verbose,
            shuffle=True,
            callbacks=callbacks,
        ).history

        # optimizer is saved like this due to tf error in serializing the model if it's a keras optimizer object
        if not isinstance(self.optimizer, str):
            self.optimizer = (
                f"{str(self.optimizer.lr)[14:19]} with "
                f"learning rate = {str(self.optimizer.lr)[-6:-1]}"
            )

        if self.plot and self.latent_dimension == 2:
            latent_activations = self.encoder.predict(x_test)
            x_test_unscaled = pd.DataFrame(
                self.scaler.inverse_transform(x_test), columns=x_test.columns, index=x_test.index,
            )

            latent_reference_col = kwargs.get("latent_plot_reference_col", None)
            if latent_reference_col is not None:
                latent_reference_col = x_test_unscaled[latent_reference_col]
            self.plot_latent_space(
                latent_activations=latent_activations, latent_reference_col=latent_reference_col,
            )
            save_or_show_plot(f"{plot_save_folder_path}\\latent_space.png")

        predictions = autoencoder.predict(x_test)
        mse = np.mean(np.power(x_test - predictions, 2), axis=1)
        self.anomaly_threshold = np.quantile(mse, self.mse_quantile)

        df_predictions = pd.DataFrame(predictions, columns=col_names_pred, index=date_index)
        df_validation = x_test

        if self.verbose:
            print(pd.DataFrame({"mse": mse}).describe())
            print(
                f"Test set MSE {self.mse_quantile * 100:3.1f}% percentile: {self.anomaly_threshold}"
            )

        if self.plot:
            self.plot_predictions(
                df_validation, df_predictions, mse=mse,
            )
            plt.suptitle("Model predictions on validation data")
            save_or_show_plot(f"{plot_save_folder_path}\\train_data_predictions.png")

            self.plot_error_distribution(mse)
            save_or_show_plot(f"{plot_save_folder_path}\\train_data_error_distribution.png")

            self.plot_learning(history)
            save_or_show_plot(f"{plot_save_folder_path}\\train_data_learning_curve.png")
        return self

    def predict(
        self, df_feature: DataFrame, plot_save_folder_path: Union[str, WindowsPath] = None,
    ) -> DataFrame:
        """
        Takes new data and uses a trained autoencoder model to predict/reconstruct it and subsequently determines
        a binary anomaly indicator for each reconstructed data point based on a fixed reconstruction error threshold.

        :param plot_save_folder_path: a valid local folder to save plots to, can be a string or a WindowsPath from pathlib
        :param df_feature: dataframe to be predicted/reconstructed by the autoencoder
        :type df_feature: Dataframe
        :param plot_save_folder_path: a path to a folder in which to save all plots produced by the autoencoder. Leave
        as None, if no saving is desired.
        :type plot_save_folder_path Union[str, WindowsPath], default=None
        :return: the resulting predictions/reconstructions of the new data in df_feature
        :rtype: DataFrame
        """
        df_feature_copy = df_feature.copy()
        cols_to_include = self.cols_to_include

        x_scaled = self.scaler.transform(df_feature_copy[cols_to_include])
        x_predicted_scaled = self.model.predict(x_scaled)

        mse = np.mean(np.power(x_predicted_scaled - x_scaled, 2), axis=1)
        df_feature_copy["mse"] = mse
        mse_threshold = self.anomaly_threshold
        df_feature_copy["anomaly_autoencoder"] = False
        anomalies = [data_point_mse > mse_threshold for data_point_mse in df_feature_copy["mse"]]
        df_feature_copy["anomaly_autoencoder"] = np.array(anomalies).astype(int)

        if self.verbose:  # pragma: no cover
            quantile = self.mse_quantile
            print("\n=== Autoencoder anomaly prediction results ===")
            print(
                f"Percentage of {len(x_scaled)} observations deemed "
                f"anomalous: {100 * np.sum(anomalies) / len(anomalies):1.2f} %"
            )
            print(f"Training data MSE,  {quantile * 100}% percentile: {mse_threshold:05.3f}")
            print(
                f"Predicted data MSE, {quantile * 100}% percentile: {np.quantile(mse, quantile):05.3f}"
            )

        if self.plot:  # pragma: no cover
            x_unscaled = self.scaler.inverse_transform(x_scaled)
            x_predicted_unscaled = self.scaler.inverse_transform(x_predicted_scaled)

            df_observed = pd.DataFrame(
                x_unscaled, columns=cols_to_include, index=df_feature_copy.index
            )
            df_predictions = pd.DataFrame(
                x_predicted_unscaled, columns=cols_to_include, index=df_feature_copy.index,
            )

            self.plot_predictions(
                df_observed,
                df_predictions,
                mse=mse,
                anomaly=df_feature_copy["anomaly_autoencoder"],
            )
            plt.suptitle("Model predictions on new data")
            save_or_show_plot(f"{plot_save_folder_path}\\predict_data_predictions.png")

            self.plot_error_distribution(mse)
            save_or_show_plot(f"{plot_save_folder_path}\\predict_data_error_distribution.png")

        return df_feature_copy[["mse", "anomaly_autoencoder"]]

    def set_scaler(self, scaler: Union[StandardScaler, MinMaxScaler]) -> None:
        """
        Helper function to set the scaler for the autoencdoer to use. This should be an already
        fitted sklearn.preprocessing scaler object.

        :param scaler: sklearn.preprocessing scaler object that has already been fitted
        """
        self.scaler = scaler

    def _build_autoencoder(self) -> Tuple[Model, Model]:
        """
        Builds the autoencoder based on the parameters given when instantiated and the call to fit.
        Returns both the full autoencoder keras model as well as the encoder part in order to obtain
        the latent space of any given input.

        :return: tuple with the full autoencoder model as well as an encoder model
        """
        self.latent_dimension = np.min(self.hidden_layers)
        hidden_layers = self.hidden_layers
        assert isinstance(self.activation_functions, str) or \
               isinstance(self.activation_functions, list), "Wrong type given as activation functions "

        if isinstance(self.activation_functions, str):
            activation_functions = [self.activation_functions for layer in hidden_layers]
            activation_functions.append("linear")
        else:
            assert len(self.activation_functions) == len(self.hidden_layers) + 1, (
                "Incorrect number of activation functions provided. The number must be |hidden_layers| + 1. "
                "Did you forget the output layer function?"
            )
            activation_functions = self.activation_functions

        original_features = Input(shape=(self.N_features,))
        x = Dense(units=hidden_layers[0], activation=activation_functions[0])(original_features)
        x = Dropout(self.dropout_fraction)(x)
        remaining_layers, remaining_activation_functions = (
            hidden_layers[1:],
            activation_functions[1:],
        )
        encoder_model = None
        for units, activation in zip(remaining_layers, remaining_activation_functions):
            if units == self.latent_dimension:
                x = Dense(units=units, activation=activation)(x)
                latent_layer = x
                encoder_model = Model(inputs=original_features, outputs=latent_layer)
            else:
                x = Dense(units=units, activation=activation)(x)
                x = Dropout(self.dropout_fraction)(x)
        reconstructed_features = Dense(self.N_features, activation=activation_functions[-1])(x)
        autoencoder = Model(inputs=original_features, outputs=reconstructed_features)
        autoencoder.compile(optimizer=self.optimizer, loss=self.loss_function)
        self.model = autoencoder
        self.encoder = encoder_model
        return self.model, self.encoder


