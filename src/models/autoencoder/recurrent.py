from typing import List, Any

import numpy as np
import pandas as pd
from keras import Input, Model
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Bidirectional
from keras.optimizers import Nadam
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

from pdm.utils.definitions import model_info
from pdm.models.autoencoder.base import AutoencoderMixin
from pdm.models.base import PlotsMixin


class LSTMAutoencoder(AutoencoderMixin, PlotsMixin):
    def __init__(
        self,
        cols_to_include=None,
        epochs: int = 250,
        batch_size: int = 128,
        optimizer: Any = Nadam(learning_rate=0.001),
    ):
        """

        :param cols_to_include: a list specifying the data columns used by the model
        :type cols_to_include: List[str], default=None
        :param epochs:
        :type epochs:
        :param batch_size:
        :type batch_size:
        :param optimizer:
        :type optimizer:
        """
        self.autoencoder_model = None
        self.scaler = None
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer

    def fit(
        self,
        x_train: DataFrame,
        x_test: DataFrame,
        scaler: StandardScaler = StandardScaler(),
        verbose: bool = True,
        plot: bool = True,
        **kwargs,
    ):
        """
        Method that handles training of the autoencoder with associated hyperparameters and architecture of the model

        :param x_train: The training dataframe
        :type x_train: DataFrame
        :param x_test: The validation/test dataframe
        :type x_test: DataFrame
        :param epochs: Number of full passes over the entire training set
        :type epochs: int, default=250
        :param batch_size: number of samples considered for each weight update in the network optimization procedure
        :type batch_size: int, default=128
        :param optimizer: the optimization algorithm to use for optimizing the weights and biases of the network
        :type optimizer: Union[str, keras.optimizer], default=Nadam(learning_rate=0.001)
        :param verbose: should the fitting process be informative or not
        :type verbose: bool, default=True
        :param plot: should the model be plotted or not
        :type plot: bool, default=True
        :return: trained model, training history, reconstruction error from the test set and two dataframes, one for
        with the validation/test dataset and with predictions from the test set. This is for compatibility with other
        functions
        :rtype: Keras model, dictionary, numpy array, DataFrame, DataFrame
        """

        if verbose:
            print("Training autoencoder model with {0} input features:".format(x_train.shape[1]))
            print(x_train.columns)

        N_features = x_train.shape[1]
        col_names = x_train.columns
        col_names_pred = col_names + "_pred"
        date_index = x_test.index

        # define feed-forward autoencoder
        input_layer = Input(shape=(N_features,))
        encoder = Dense(16, activation="tanh")(input_layer)
        encoder = Dense(15, activation="tanh")(encoder)
        encoder = Dense(14, activation="tanh")(encoder)
        latent = Dense(2, activation="tanh")(encoder)
        decoder = Dense(14, activation="tanh")(latent)
        decoder = Dense(15, activation="tanh")(decoder)
        decoder = Dense(16, activation="tanh")(decoder)
        output_layer = Dense(N_features, activation="linear")(decoder)
        autoencoder = Model(input_layer, output_layer)
        if verbose:
            autoencoder.summary()

        autoencoder.compile(optimizer=self.optimizer, loss="mse")
        history = autoencoder.fit(
            x_train,
            x_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(x_test, x_test),
            verbose=verbose,
            use_multiprocessing=True,
            shuffle=True,
        ).history

        # plot the latent dimension of the network # TODO make this available for the prediction fun as well
        if plot and latent.shape[1] == 2:
            # extract the latent layer activations
            idx_latent = int(latent.name[6])
            latent_output = autoencoder.layers[idx_latent].output
            activation_model = Model(inputs=input_layer, outputs=latent_output)
            activations = activation_model.predict(x_test)

            # use pump rotation to color the points, so we need to unscale it
            x_test_unscaled = pd.DataFrame(
                scaler.inverse_transform(x_test), columns=x_test.columns, index=x_test.index
            )

            # make the plot
            cmap = sns.cubehelix_palette(gamma=0.7, reverse=True, as_cmap=True)
            fig, ax = plt.subplots()
            points = ax.scatter(
                x=activations[:, 0],
                y=activations[:, 1],
                c=x_test_unscaled["pump_rotation"],
                cmap=cmap,
            )
            cbar = fig.colorbar(points)
            cbar.ax.get_yaxis().labelpad = 20
            cbar.ax.set_ylabel("Pump rotation [rpm]", rotation=270)
            plt.title("Latent layer", loc="left", fontsize=18)
            plt.title(
                "Activation function: {}".format(latent.op.type.lower()),
                loc="right",
                fontsize=10,
                color="grey",
            )
            plt.xlabel("Unit 1")
            plt.ylabel("Unit 2")
            plt.show()

        predictions = autoencoder.predict(x_test)
        mse = np.mean(np.power(x_test - predictions, 2), axis=1)

        df_predictions = pd.DataFrame(predictions, columns=col_names_pred, index=date_index)
        df_validation = x_test

        if verbose:
            print(pd.DataFrame({"mse": mse}).describe())
            print("Test set MSE 0.5% percentile (5 out of 1000):", np.quantile(mse, 0.995))

        # TODO the AE plotting should happen inside "train_autoencoder" and "train_lstm" - as in the pwlf and dbscan
        if plot:
            self.plot_predictions(
                df_validation,
                df_predictions,
                cols_to_plot=[
                    "pump_rotation",
                    "pump_rotation_pred",
                    "flush_indicator",
                    "flush_indicator_pred",
                ],
                mse=mse,
            )
            self.plot_error_distribution(mse)
            self.plot_learning(history)

        # model_output = autoencoder, history, mse, df_validation, df_predictions
        # autoencoder, history, mse, df_validation, df_predictions = model_output

        self.model["autoencoder"] = model_info(autoencoder, scaler)

    def predict(self, df_feature: DataFrame, cols_to_include: List[str] = None):
        """
        Takes new data and uses a trained autoencoder model to predict/reconstruct it and subsequently determines
        a binary anomaly indicator for each reconstructed data point based on a fixed reconstruction error threshold.

        :param df_feature: dataframe to be predicted/reconstructed by the autoencoder
        :type df_feature: Dataframe
        :param cols_to_include: the data columns to include - must be the same that went into training the model
        :type cols_to_include: List[str], default=None
        :return: the resulting predictions/reconstructions of the new data in df_feature
        :rtype: DataFrame
        """

        col_names = df_feature[cols_to_include].columns
        model_type = "autoencoder_tags"
        if self.lstm_model:
            window_size = self.model[model_type].input_shape[1]
            x_scaled = self.scaler[model_type].transform(df_feature[cols_to_include], copy=True)
            x_scaled = self.make_sliding_windows(x_scaled, window_size)
            try:
                x_predicted_scaled = self.model[model_type].predict(x_scaled)
            except ValueError:
                print(
                    "Error: Check that the trained model is of the same model type - "
                    "e.g. not an ordinary autoencoder"
                )
                raise ValueError
            x_predicted_scaled = self.concat_sliding_windows(x_predicted_scaled)
            x_scaled = self.concat_sliding_windows(x_scaled)

            # to accommodate the size of the sliding windows the first n few observations may be discarded
            N_rows_to_discard = len(df_feature) - len(x_scaled)
            df_feature = df_feature[N_rows_to_discard:]
        else:
            x_scaled = self.scaler[model_type].transform(df_feature[cols_to_include], copy=True)
            try:
                x_predicted_scaled = self.model[model_type].predict(x_scaled)
            except ValueError:
                print(
                    "Error: Check that the trained model is of the same model type - e.g. not an LSTM model"
                )
                raise ValueError

        mse = np.mean(np.power(x_predicted_scaled - x_scaled, 2), axis=1)
        name_suffix = model_type
        df_feature["mse_" + name_suffix] = mse
        mse_threshold = self._get_anomaly_threshold(model_type)["mse_threshold"]
        df_feature["anomaly_" + name_suffix] = False
        anomalies = [
            data_point_mse > mse_threshold for data_point_mse in df_feature["mse_" + name_suffix]
        ]
        df_feature["anomaly_" + name_suffix] = anomalies
        df_feature["anomaly_ae_binary"] = np.array(anomalies).astype(int)

        if self.plot:  # pragma: no cover
            quantile = 0.995
            print(
                "Training data MSE, {0}% percentile: {1:05.3f}".format(
                    quantile * 100, mse_threshold
                )
            )
            print(
                "Predicted data MSE, {0}% percentile: {1:05.3f}".format(
                    quantile * 100, np.quantile(mse, quantile)
                )
            )
            print(pd.DataFrame({"mse": mse}).describe())

            x_unscaled = self.scaler[model_type].inverse_transform(x_scaled, copy=True)
            x_predicted_scaled = self.scaler[model_type].inverse_transform(
                x_predicted_scaled, copy=True
            )

            df_observed = pd.DataFrame(x_unscaled, columns=col_names, index=df_feature.index)
            df_predictions = pd.DataFrame(
                x_predicted_scaled, columns=col_names, index=df_feature.index
            )

            cols_to_plot = [
                "pump_rotation",
                "pump_rotation_pred",
                "pump_rotation_other",
                "pump_rotation_other_pred",
                "kv_flow",
                "kv_flow_pred",
                "leje_bs_vibr_x",
                "leje_bs_vibr_x_pred",
                "flush_indicator",
                "flush_indicator_pred",
            ]

            self.plot_predictions(
                df_observed,
                df_predictions,
                cols_to_plot=cols_to_plot,
                mse=mse,
                anomaly_ae_binary=df_feature["anomaly_ae_binary"],
            )
            self.plot_error_distribution(
                mse=mse,
                component_name=self.model_tags["component-name"],
                training_period=self.model_tags["training-period"],
            )

        return df_feature[["mse_autoencoder_tags", "anomaly_ae_binary"]]

    # def train_lstm(
    #     self,
    #     x_train: DataFrame,
    #     x_test: DataFrame,
    #     window_size: int = 10,
    #     epochs: int = 50,
    #     batch_size: int = 128,
    #     optimizer: str = "adam",
    # ):
    #
    #     assert window_size > 0
    #
    #     if self.verbose:
    #         print("Training LSTM autoencoder with {0} input features:".format(x_train.shape[1]))
    #         print(x_train.columns)
    #
    #     N_features = x_train.shape[1]
    #     col_names = x_train.columns
    #     col_names_pred = col_names + "_pred"
    #
    #     x_train = np.array(x_train)
    #     x_test = np.array(x_test)
    #     x_train = self.make_sliding_windows(x_train, window_size=window_size)
    #     x_test = self.make_sliding_windows(x_test, window_size=window_size)
    #
    #     # define seq2seq autoencoder model
    #     input_layer = Input(shape=(window_size, N_features))
    #     encoder = Bidirectional(
    #         LSTM(6, activation="relu", return_sequences=True),
    #         input_shape=(window_size, N_features),
    #         merge_mode="concat",
    #     )(input_layer)
    #     encoder = Bidirectional(
    #         LSTM(3, activation="relu", return_sequences=False), merge_mode="concat"
    #     )(encoder)
    #     latent_layer = RepeatVector(window_size)(encoder)
    #     decoder = Bidirectional(
    #         LSTM(3, activation="relu", return_sequences=True), merge_mode="concat"
    #     )(latent_layer)
    #     decoder = Bidirectional(
    #         LSTM(6, activation="relu", return_sequences=True), merge_mode="concat"
    #     )(decoder)
    #     output_layer = TimeDistributed(Dense(N_features))(decoder)
    #
    #     autoencoder = Model(input_layer, output_layer)
    #     autoencoder.summary()
    #
    #     autoencoder.compile(optimizer=optimizer, loss="mse")
    #     history = autoencoder.fit(
    #         x_train,
    #         x_train,
    #         epochs=epochs,
    #         batch_size=batch_size,
    #         verbose=self.verbose,
    #         validation_data=(x_test, x_test),
    #         shuffle=True,
    #     ).history
    #
    #     predictions = self.concat_sliding_windows(autoencoder.predict(x_test, verbose=self.verbose))
    #     df_validation = pd.DataFrame(self.concat_sliding_windows(x_test), columns=col_names)
    #     df_predictions = pd.DataFrame(predictions, columns=col_names_pred)
    #     mse = np.mean(np.power(np.array(df_predictions) - np.array(df_validation), 2), 1)
    #
    #     if self.verbose:
    #         print(pd.DataFrame({"mse": mse}).describe())
    #         print("Test set MSE 0.5% quantile (5 out of 1000):", np.quantile(mse, 0.995))
    #
    #     return autoencoder, history, mse, df_validation, df_predictions
