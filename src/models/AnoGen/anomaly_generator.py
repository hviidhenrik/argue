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

from src.models.AnoGen.utility_functions import fit_VAE


class AnomalyDetectorAutoencoder:
    def __init__(self,
                 intermediate_dim,
                 latent_dim
                 ):
        self.latent_dim = latent_dim
        self.intermediate_dim = intermediate_dim
        self.model = None
        self.anomaly_threshold = None
        self.mse_train = None
        self.mse_val = None
        self.epochs = None
        self.early_stopping = None
        self.loss = None
        self.batch_size = None
        self.activation = None
        self.scaler = None

    def fit(self,
            df_x_train,
            df_x_val,
            epochs=200,
            early_stopping=True,
            loss="mse",
            batch_size=128,
            activation="tanh",
            plot_history=False,
            plot_latent=False):
        """
        Trains an ordinary autoencoder for anomaly detection
        """

        self.scaler = MinMaxScaler()
        x_train_scaled = self.scaler.fit_transform(df_x_train)
        x_val_scaled = self.scaler.transform(df_x_val)

        callbacks = []
        if early_stopping:
            callbacks.append(EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True))

        # Specify hyperparameters
        original_dim = x_train_scaled.shape[1]

        # Encoder
        x = Input(shape=(original_dim,))
        # h = Dropout(0.1)(x)
        h = Dense(self.intermediate_dim, activation=activation)(x)
        # h = BatchNormalization()(h)
        h = Dense(self.intermediate_dim - 2, activation=activation)(h)
        # h = BatchNormalization()(h)
        # h = Dropout(0.1)(h)
        h = Dense(self.intermediate_dim - 4, activation=activation)(h)

        # bottleneck
        latent = Dense(self.latent_dim, activation=activation)(h)

        # This defines the Encoder which takes noise and input, and outputs
        # the latent variable z
        encoder = Model(inputs=x, outputs=latent)

        # Decoder is MLP specified as single Keras Sequential Layer
        decoder = Sequential([
            Dense(self.intermediate_dim - 4, input_dim=self.latent_dim, activation=activation),
            # Dropout(0.1),
            # BatchNormalization(),
            Dense(self.intermediate_dim - 2, input_dim=self.latent_dim, activation=activation),
            # BatchNormalization(),
            Dense(self.intermediate_dim, input_dim=self.latent_dim, activation=activation),
            # Dropout(0.1),
            Dense(original_dim, activation='tanh')
        ])

        x_pred = decoder(latent)

        autoencoder = Model(inputs=x, outputs=x_pred, name='ae')
        autoencoder.compile(optimizer='adam', loss=loss)

        hist = autoencoder.fit(
            x_train_scaled,
            x_train_scaled,
            shuffle=False,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_val_scaled, x_val_scaled),
            verbose=2,
            callbacks=callbacks
        )

        preds_train = autoencoder.predict(x_train_scaled)
        preds_val = autoencoder.predict(x_val_scaled)
        mse_train = np.mean(np.power(x_train_scaled - preds_train, 2), axis=1)
        mse_val = np.mean(np.power(x_val_scaled - preds_val, 2), axis=1)

        if plot_latent:
            z_train = encoder.predict(x_train_scaled)
            title_latent = "Autoencoder latent space (training data)"
            if self.latent_dim > 2:
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

        if plot_history:
            # Training loss plot
            fig, ax = plt.subplots()
            hist_df = pd.DataFrame(hist.history)
            hist_df.plot(ax=ax)
            plt.suptitle("Autoencoder learning curve")
            ax.set_ylabel('Loss')
            ax.set_xlabel('# epochs')
            plt.show()

        self.model = autoencoder
        self.mse_train = mse_train
        self.mse_val = mse_val
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.loss = loss
        self.batch_size = batch_size
        self.activation = activation
        return self

    def predict(self, df_x_predict, anomaly_threshold):
        x_predict_scaled = self.scaler.transform(df_x_predict)
        x_predicted_scaled = self.model.predict(x_predict_scaled)
        mse_predicted = np.mean(np.power(np.array(x_predict_scaled) - x_predicted_scaled, 2), axis=1)
        anomalies = [data_point_mse > anomaly_threshold for data_point_mse in mse_predicted]
        anomalies = np.array(anomalies).astype(int)
        df_anomalies = pd.DataFrame({"AE_mse": mse_predicted,
                                     "AE_anomaly": anomalies})
        self.anomaly_threshold = anomaly_threshold
        return df_anomalies

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


class AnomalyGenerator:
    def __init__(self,
                 vae_intermediate_dim: int = 12,
                 vae_latent_dim: int = 2):
        self.feature_cols = None
        self.vae_intermediate_dim = vae_intermediate_dim
        self.vae_latent_dim = vae_latent_dim
        self.vae_validation_size = None
        self._x_train = None
        self._x_val = None
        self._scaler = None
        self._vae_latent_plot_pca = None
        self.latent_stddev = None
        self.activation = None
        self.encoder = None
        self.decoder = None
        self.model = None
        self._df_vae_latent_space = None
        self.clustering = None
        self.dbscan_epsilon = None
        self.latent_cols = None

    def fit(self,
            df,
            vae_validation_size: float = 0.1,
            latent_stddev: float = 0.005,
            dbscan_epsilon: float = 0.3,  # 0.3 for 2 dim vae
            activation: str = "elu",
            epochs: int = 200,
            early_stopping: bool = True,
            kl_warmup: int = 30,
            plot_history: bool = False):

        self._df_train = df.copy()

        # split in train and validation set
        x_train_unscaled, x_val_unscaled = train_test_split(self._df_train,
                                                            test_size=vae_validation_size,
                                                            shuffle=False)

        # scale train and validation data
        self._scaler = MinMaxScaler()
        x_train_scaled = self._scaler.fit_transform(x_train_unscaled)
        x_val_scaled = self._scaler.transform(x_val_unscaled)

        # TODO make a VAE class
        encoder, decoder, vae = fit_VAE(x_train_scaled,
                                        x_val_scaled,
                                        intermediate_dim=self.vae_intermediate_dim,
                                        latent_dim=self.vae_latent_dim,
                                        batch_size=128,
                                        epochs=epochs,
                                        early_stopping=early_stopping,
                                        kl_warmup=kl_warmup,
                                        latent_stddev=latent_stddev,
                                        plot_history=plot_history,
                                        activation=activation)

        self.latent_cols = [f"z{dim}" for dim in range(self.vae_latent_dim)]
        self._df_vae_latent_space = pd.DataFrame(
            encoder.predict(
                self._scaler.fit_transform(self._df_train)),  # refit scaler to whole training set including validation
            columns=self.latent_cols
        )

        # fit clustering model to recognize the learned latent space and filter away samples that are
        # overlapping with this learned normal condition
        self.clustering = DBSCANClustering(cols_to_include=self._df_vae_latent_space.columns.values,
                                           epsilon=dbscan_epsilon,
                                           guide_search_for_eps=False,
                                           plot=False
                                           ).fit(self._df_vae_latent_space)
        self.decoder = decoder
        self.feature_cols = df.columns.values
        self.latent_stddev = latent_stddev
        self.activation = activation
        self.encoder = encoder
        self.decoder = decoder
        self.model = vae
        self.dbscan_epsilon = dbscan_epsilon
        return self

    def generate_anomalies(self,
                           N_samples: int = 10,
                           z_min: float = -1,
                           z_max: float = 1,
                           domain_filter: Dict[str, List[float]] = None):
        """
        Sample anomaly points from the VAE latent space based on the training data.
        """
        samples_accepted = 0
        df_anomalies_decoded_final = pd.DataFrame(columns=self.feature_cols)
        df_anomalies_latent_final = pd.DataFrame(columns=self.latent_cols)
        while samples_accepted < N_samples:
            # sample 10*N_samples due to many being rejected and faster processing when done in bigger batches
            df_anomalies_latent = pd.DataFrame(np.random.uniform(z_min, z_max, (10 * N_samples, self.vae_latent_dim)),
                                               columns=self.latent_cols)
            df_latent_samples_decoded = pd.DataFrame(
                self._scaler.inverse_transform(
                    self.decoder.predict(df_anomalies_latent)),
                columns=self.feature_cols
            )

            # filter unrealistic anomalies away based on domain knowledge and observed ranges in training set
            df_to_filter = pd.concat([df_anomalies_latent, df_latent_samples_decoded], axis=1)
            df_sampled_anomalies = self._filter_sampled_anomalies(df_to_filter, domain_filter)
            # df_sampled_anomalies = df_to_filter


            # print(samples_accepted)
            # if df_sampled_anomalies.shape[0] == 0:
            #     continue

            # use the fitted clustering model to filter away samples that are in the part of latent space
            # that represents the normal condition learned by the VAE
            df_outside_nominal = self.clustering.predict(df_sampled_anomalies)
            df_sampled_anomalies = pd.concat([df_sampled_anomalies, df_outside_nominal], axis=1)
            df_sampled_anomalies = df_sampled_anomalies.loc[df_sampled_anomalies["anomaly_dbscan"] == 1, :]
            df_sampled_anomalies = df_sampled_anomalies.drop("anomaly_dbscan", axis=1)

            df_sampled_anomalies_decoded = df_sampled_anomalies.iloc[:, self.vae_latent_dim:].reset_index(drop=True)
            df_sampled_anomalies_latent = df_sampled_anomalies.iloc[:, :self.vae_latent_dim].reset_index(drop=True)
            df_anomalies_decoded_final = pd.concat([df_anomalies_decoded_final, df_sampled_anomalies_decoded], axis=0)
            df_anomalies_latent_final = pd.concat([df_anomalies_latent_final, df_sampled_anomalies_latent], axis=0)
            samples_accepted = df_anomalies_decoded_final.shape[0]

        # discard abundant samples
        df_anomalies_decoded_final = df_anomalies_decoded_final.iloc[:N_samples, ]
        df_anomalies_latent_final = df_anomalies_latent_final.iloc[:N_samples, ]
        return df_anomalies_decoded_final, df_anomalies_latent_final

    def _filter_sampled_anomalies(self,
                                  df_latent_and_reconstructions,
                                  domain_filter=None):
        """
        Filters generated anomalies based on a dataframe of nominal values for column and optionally a
        filter based on prior domain knowledge or physical constraints. The limits found in these two are combined
        to find the minimal and maximally allowed limits for the generated anomalies to look realistic.

        :param df_latent_and_reconstructions: dataframe of latent samples and their reconstructed values in feature space. Must have
        the latent coords in the leftmost columns.
        :type df_latent_and_reconstructions: DataFrame
        :param domain_filter: a dictionary like {"feature_1": [0, 15000], "feature_2": [0, 1], ...}
        :type domain_filter: Dict[str, List]
        :return: a dataframe with only the accepted anomalies that are within the found limits
        :rtype: DataFrame
        """

        df_filtered = df_latent_and_reconstructions.copy()
        nominal_max = self._df_train.max()
        nominal_min = self._df_train.min()
        if domain_filter is not None:
            # find common cols in both domain_filter and _x_train and retain ordering
            common_cols = [col for col in list(domain_filter.keys()) if col in self.feature_cols]
            # overrule limits found in train set by limits from provided domain knowledge filter
            for col in common_cols:
                # if upper or lower bound win domain filter is undefined, simply use bound from train set
                domain_min = nominal_min[col] if domain_filter[col][0] is None else domain_filter[col][0]
                domain_max = nominal_min[col] if domain_filter[col][1] is None else domain_filter[col][1]
                nominal_min[col] = np.min((domain_min, nominal_min[col]))
                nominal_max[col] = np.max((domain_max, nominal_max[col]))
        for col in self.feature_cols:
            df_filtered = df_filtered[(nominal_min[col] < df_filtered[col]) & (df_filtered[col] < nominal_max[col])]
        return df_filtered

    def get_vae_latent_space(self):
        return self._df_vae_latent_space

    def plot_vae_latent(self, color_by_columns=None, save=False, show=True, ax=None):
        if color_by_columns is None:
            color_by_columns = [self._df_train.columns[0]]
        elif not isinstance(color_by_columns, list):
            if color_by_columns.lower() == "all":
                color_by_columns = self._df_train.columns.values
            else:
                color_by_columns = [color_by_columns]  # if only a single string is received like "kv_flow"
        df = self._df_vae_latent_space.copy()
        pca_reduce = False
        title_latent = "AnoGen VAE training latent space"
        if self.vae_latent_dim > 2:
            pca = PCA(n_components=2).fit(df)
            df = pd.DataFrame(pca.fit_transform(df))
            var_expl = 100 * pca.explained_variance_ratio_.sum()
            title_latent = title_latent + "\nPCA reduced (var explained: {0:4.0f})%".format(var_expl)
            pca_reduce = True
            self._vae_latent_plot_pca = pca

        for coloring_col in color_by_columns:
            plt.scatter(df.iloc[:, 0], df.iloc[:, 1],
                        c=self._df_train[coloring_col], cmap='jet', s=10)
            plt.xlabel("PC1" if pca_reduce else "z0")
            plt.ylabel("PC2" if pca_reduce else "z1")
            clb = plt.colorbar()
            clb.set_label(coloring_col, rotation=0, labelpad=-30, y=1.05)
            plt.title(title_latent)
            if save:
                plt.savefig('VAE_latent_[{}].png'.format(coloring_col))
            if show:
                plt.show()

    def save(self):
        # TODO find out how to save the VAE with its custom layers and implement this
        pass

    def load(self):
        pass


class AnomalyEvaluator:
    def __init__(self):
        self.df_samples_latent_and_x_and_preds = None
        self.df_sample_latent_space = None
        self.df_samples_reconstructions = None
        self.df_vae_latent_space = None
        self.df_vae_train_data = None
        self.vae_latent_pca = None
        self.vae_latent_dim = None
        self.metrics = None

    def _append_anomalies(self, df_nominal, df_anomalies):
        """
        Takes a normal df and an anomaly df and makes an extra column in each, indicating
        if each datapoint is anomalous or nominal. After this, it concatenates them vertically
        and makes sure the timeindex is correct by simply continuing the nominal df's index into
        the anomalous one by incrementing it appropriately

        :param df_nominal: the dataframe with nominal/normal data to have anomalies appended to it
        :type df_nominal: DataFrame
        :param df_anomalies: the dataframe with anomalies to append to df
        :type df_anomalies: DataFrame
        :return: a dataframe containing the nominal data with anomalous data concatted vertically as the last rows
        :rtype: DataFrame
        """

        assert list(df_nominal.columns) == list(df_anomalies.columns)

        df_copy = df_nominal.copy()
        df_anomalies_copy = df_anomalies.copy()

        df_copy = df_copy.assign(synthetic_anomaly=0)
        df_anomalies_copy = df_anomalies_copy.assign(synthetic_anomaly=1)
        N_anomaly = df_anomalies_copy.shape[0]

        if not type(df_nominal.index) == type(df_anomalies.index):
            time_index = pd.to_datetime(df_copy.index)
            time_increment = time_index[-1] - time_index[-2]
            anomaly_index = [time_index[-1] + time_increment]
            for i in range(N_anomaly - 1):
                anomaly_index.append(anomaly_index[i] + time_increment)

        df_anomalies_copy.index = anomaly_index
        df_concatted = pd.concat([df_copy, df_anomalies_copy])
        df_concatted.index = pd.to_datetime(df_concatted.index)
        return df_concatted

    def _compute_performance_metrics(self, anomaly_truths, anomaly_predicted):
        """
        Computes metrics to evaluate the ability and skill of an anomaly detection algorithm
        given known anomalies and nominal observations.

        For the confusion matrix, the following values can be obtained:
                           ACTUAL
                         |    1   |   0
                ----------------------------
        PREDICTED     1  |    TP    |  FP
                ----------------------------
                      0  |    FN    |  TN

        FPR                        = FP / (FP + TN)
        sensitivity (TPR)          = TP / (TP + FN)
        precision (PPV)            = TP / (TP + FP)
        False discovery rate (FDR) = FP / (FP + TP)


        """
        # cnf_matrix = confusion_matrix(anomaly_truths, anomaly_predictions)
        TN, FP, FN, TP = confusion_matrix(anomaly_truths, anomaly_predicted).ravel()
        N_total = anomaly_truths.shape[0]
        N_positives = anomaly_truths.sum()
        N_negatives = N_total - N_positives

        # Sensitivity, hit rate, recall, or true positive rate
        TPR = TP / N_positives
        # Specificity or true negative rate
        TNR = TN / N_negatives
        # Precision or positive predictive value
        PPV = TP / (TP + FP)
        # Negative predictive value
        NPV = TN / (TN + FN)
        # Fall out or false positive rate
        FPR = FP / N_negatives
        # False negative rate
        FNR = FN / N_positives
        # False discovery rate
        FDR = FP / (FP + TP)
        # accuracy
        ACC = (TP + TN) / (N_positives + N_negatives)
        # balanced accuracy
        balanced_ACC = (TPR + TNR) / 2
        # balanced false rate (BFR)
        BFR = (FPR + FNR) / 2  # could consider making a weighted average here with higher weight on e.g. FPR
        # F1 score
        F1 = 2 * TP / (2 * TP + FP + FN)
        # Area under curve
        AUC = roc_auc_score(anomaly_truths, anomaly_predicted)
        # "mean Average Precision (mAP)"
        AP = average_precision_score(anomaly_truths, anomaly_predicted)
        self.metrics = {"TPR": TPR, "TNR": TNR, "precision:": PPV, "NPV": NPV, "FPR": FPR, "FNR": FNR,
                        "FDR": FDR, "accuracy": ACC, "balanced_ACC": balanced_ACC, "BFR": BFR,
                        "F1": F1, "AUC": AUC, "AP": AP}

    # def collect_visualization_dataframe(self,
    #                                     df_anomalies_latent,
    #                                     df_anomalies_decoded,
    #                                     df_samples_anomaly_predictions,
    #                                     df_vae_latent_space,
    #                                     df_vae_train_data,
    #                                     df_test_data,
    #                                     df_test_anom_preds,
    #                                     anomaly_threshold,
    #                                     quantile):
    #     assert df_anomalies_latent.shape[0] == df_anomalies_decoded.shape[0]
    #     assert df_anomalies_decoded.shape[0] == df_samples_anomaly_predictions.shape[0]
    #
    #     # collect all information about the sampled anomalies in one df
    #     df_samples_latent_and_x_and_preds = pd.concat([df_anomalies_latent,
    #                                                    df_anomalies_decoded,
    #                                                    df_samples_anomaly_predictions],
    #                                                   axis=1)
    #     df_test_x_and_preds = pd.concat([df_test_data, df_test_anom_preds], axis=1)
    #     df_samples_x_and_preds = pd.concat([df_anomalies_decoded, df_samples_anomaly_predictions], axis=1)
    #     df_test_and_samples_and_preds = self._append_anomalies(df_test_x_and_preds, df_samples_x_and_preds)
    #
    #     self.df_samples_latent_and_x_and_preds = df_samples_latent_and_x_and_preds
    #     self.df_sample_latent_space = df_anomalies_latent
    #     self.df_anomalies_decoded = df_anomalies_decoded
    #     self.df_vae_latent_space = df_vae_latent_space
    #     self.df_vae_train_data = df_vae_train_data
    #     self.df_test_data = df_test_data
    #     self.df_test_anom_preds = df_test_anom_preds
    #     self.vae_latent_dim = df_vae_latent_space.shape[1]
    #     self.df_test_and_samples_and_preds = df_test_and_samples_and_preds
    #     self.anomaly_threshold = anomaly_threshold
    #     self.quantile = quantile
    #     # TODO should do PCA of latent space here if dim > 2

    def plot_vae_latent(self, color_by_columns=None, save=False, show=True, ax=None):
        if color_by_columns is None:
            color_by_columns = [self.df_vae_train_data.columns[0]]
        elif type(color_by_columns) != list:
            if color_by_columns.lower() == "all":
                color_by_columns = self.df_vae_train_data.columns.values
            else:
                color_by_columns = [color_by_columns]  # if only a single string is received like "kv_flow"
        df = self.df_vae_latent_space.copy()
        pca_reduce = False
        title_latent = "AnoGen VAE training latent space"
        if self.vae_latent_dim > 2:
            pca = PCA(n_components=2).fit(df)
            df = pd.DataFrame(pca.fit_transform(df))
            var_expl = 100 * pca.explained_variance_ratio_.sum()
            title_latent = title_latent + "\nPCA reduced (var explained: {0:4.0f})%".format(var_expl)
            pca_reduce = True
            self.vae_latent_pca = pca

        for coloring_col in color_by_columns:
            plt.scatter(df.iloc[:, 0], df.iloc[:, 1],
                        c=self.df_vae_train_data[coloring_col], cmap='jet', s=10)
            plt.xlabel("PC1" if pca_reduce else "z0")
            plt.ylabel("PC2" if pca_reduce else "z1")
            clb = plt.colorbar()
            clb.set_label(coloring_col, rotation=0, labelpad=-30, y=1.05)
            plt.title(title_latent)
            if save:
                plt.savefig('VAE_latent_[{}].png'.format(coloring_col))
            if show:
                plt.show()

    def plot_vae_latent_with_samples(self, color_by_columns=None, save=False, show=True):
        if color_by_columns is None:
            color_by_columns = [self.df_vae_train_data.columns[0]]
        elif type(color_by_columns) != list:
            if color_by_columns.lower() == "all":
                color_by_columns = self.df_vae_train_data.columns.values
            else:
                color_by_columns = [color_by_columns]  # if only a single string is received like "kv_flow"
        df = self.df_vae_latent_space.copy()
        df_samples = self.df_sample_latent_space.copy()
        pca_reduce = False
        title_latent = "AnoGen VAE training latent space"
        if self.vae_latent_dim > 2:
            pca = PCA(n_components=2).fit(df)
            df = pd.DataFrame(pca.fit_transform(df))
            df_samples = pd.DataFrame(pca.transform(df_samples))
            var_expl = 100 * pca.explained_variance_ratio_.sum()
            title_latent = title_latent + "\nPCA reduced (var explained: {0:4.0f})%".format(var_expl)
            pca_reduce = True
            self.vae_latent_pca = pca
        for coloring_col in color_by_columns:
            plt.scatter(df.iloc[:, 0], df.iloc[:, 1],
                        c=self.df_vae_train_data[coloring_col], cmap='jet', s=10)
            plt.scatter(df_samples.iloc[:, 0], df_samples.iloc[:, 1],
                        s=13, c=self.df_samples_reconstructions[coloring_col],
                        marker="^", label="Sampled anomaly point")
            plt.xlabel("PC1" if pca_reduce else "z0")
            plt.ylabel("PC2" if pca_reduce else "z1")
            clb = plt.colorbar()
            clb.set_label(coloring_col, rotation=0, labelpad=-30, y=1.05)
            plt.title(title_latent)
            if save:
                plt.savefig('VAE_latent_[{}].png'.format(coloring_col))
            if show:
                plt.show()

    def plot_vae_latent_samples_by_anomaly_prediction(self):
        self.plot_vae_latent(show=False)
        df = self.df_samples_latent_and_x_and_preds.copy()
        latent_cols = list(df.columns[:self.vae_latent_dim])
        df_samples_detected = df.loc[df["AE_anomaly"] == 1, latent_cols]
        df_samples_not_detected = df.loc[df["AE_anomaly"] == 0, latent_cols]

        if self.vae_latent_dim > 2:
            df_samples_detected = pd.DataFrame(self.vae_latent_pca.transform(df_samples_detected))
            df_samples_not_detected = pd.DataFrame(self.vae_latent_pca.transform(df_samples_not_detected))

        plt.scatter(df_samples_detected.iloc[:, 0], df_samples_detected.iloc[:, 1],
                    s=12, c="blue", marker="^", label="Anomaly sample detected")
        plt.scatter(df_samples_not_detected.iloc[:, 0], df_samples_not_detected.iloc[:, 1],
                    s=14, c="red", marker="s", label="Anomaly sample not detected")
        plt.legend()
        plt.show()

    def plot_anomaly_time_series(self, N_nominal_to_show=200, show=True):
        df = self.df_test_and_samples_and_preds
        N_samples = self.df_samples_reconstructions.shape[0]
        idx_to_plot_after = -N_samples - N_nominal_to_show
        cols_to_plot = ["AE_mse", "AE_anomaly", "synthetic_anomaly"]
        axes = df[cols_to_plot].iloc[idx_to_plot_after:, :].plot(subplots=True, layout=(len(cols_to_plot), 1))
        plt.suptitle("Anomaly detector predictions\nSampled anomalies after black line yo")
        for i, c in enumerate(axes):
            for ax in c:
                if i == 0:
                    ax.axhline(y=self.anomaly_threshold, color="red", linestyle="--", label="Anomaly threshold")
                ax.axvline(x=df.index[-N_samples], color='black', linestyle='--')
                ax.legend(loc="upper left")
        if show:
            plt.show()
        return axes

    def print_metrics(self, combine_with_test_set=False):
        if combine_with_test_set:
            anomaly_preds = self.df_test_and_samples_and_preds["AE_anomaly"]
            anomaly_truths = self.df_test_and_samples_and_preds["synthetic_anomaly"]
        else:
            anomaly_preds = list(self.df_samples_latent_and_x_and_preds["AE_anomaly"])
            anomaly_truths = [1 for i in range(len(anomaly_preds))]

        self._compute_performance_metrics(anomaly_truths, anomaly_preds)
        print("\n\n==== Anomaly Detection performance summary ====")
        print("Anomaly detection MSE threshold: {0:1.4f}, train-set quantile: {1}\n".format(self.anomaly_threshold,
                                                                                            self.quantile))
        print("False positive rate: {0:1.2f}".format(self.metrics["FPR"]))
        print("False negative rate: {0:1.2f}".format(self.metrics["FNR"]))
        print("Balanced false rate: {0:1.2f}".format(self.metrics["BFR"]))
        print("AUC score: {0:1.2f}".format(self.metrics["AUC"]))
        print("Average precision: {0:1.2f}".format(self.metrics["AP"]))
        print("F1: {0:1.2f}".format(self.metrics["F1"]))
        print("\nConfusion matrix:\n", confusion_matrix(anomaly_truths, anomaly_preds))
        print("====================================================")

    def plot_all(self):
        """
        Plot all the plots as subplots to show at the same time for better overview. Easier said that done...
        :return:
        """
        pass


class Visualizer:
    def plot_vae_latent(self, color_by_columns=None, save=False, show=True, ax=None):
        if color_by_columns is None:
            color_by_columns = [self.df_vae_train_data.columns[0]]
        elif type(color_by_columns) != list:
            if color_by_columns.lower() == "all":
                color_by_columns = self.df_vae_train_data.columns.values
            else:
                color_by_columns = [color_by_columns]  # if only a single string is received like "kv_flow"
        df = self.df_vae_latent_space.copy()
        pca_reduce = False
        title_latent = "AnoGen VAE training latent space"
        if self.vae_latent_dim > 2:
            pca = PCA(n_components=2).fit(df)
            df = pd.DataFrame(pca.fit_transform(df))
            var_expl = 100 * pca.explained_variance_ratio_.sum()
            title_latent = title_latent + "\nPCA reduced (var explained: {0:4.0f})%".format(var_expl)
            pca_reduce = True
            self.vae_latent_pca = pca

        for coloring_col in color_by_columns:
            plt.scatter(df.iloc[:, 0], df.iloc[:, 1],
                        c=self.df_vae_train_data[coloring_col], cmap='jet', s=10)
            plt.xlabel("PC1" if pca_reduce else "z0")
            plt.ylabel("PC2" if pca_reduce else "z1")
            clb = plt.colorbar()
            clb.set_label(coloring_col, rotation=0, labelpad=-30, y=1.05)
            plt.title(title_latent)
            if save:
                plt.savefig('VAE_latent_[{}].png'.format(coloring_col))
            if show:
                plt.show()


class DetectorEvaluator:

    def __init__(self):
        self.metrics = None

    def _compute_performance_metrics(self, anomaly_truths, anomaly_predictions, threshold):
        """
        Computes metrics to evaluate the ability and skill of an anomaly detection algorithm
        given known anomalies and nominal observations.

        For the confusion matrix, the following values can be obtained:
                           ACTUAL
                         |    1   |   0
                ----------------------------
        PREDICTED     1  |    TP    |  FP
                ----------------------------
                      0  |    FN    |  TN

        FPR                        = FP / (FP + TN)
        sensitivity (TPR)          = TP / (TP + FN)
        precision (PPV)            = TP / (TP + FP)
        False discovery rate (FDR) = FP / (FP + TP)


        """
        # cnf_matrix = confusion_matrix(anomaly_truths, anomaly_predictions)
        TN, FP, FN, TP = confusion_matrix(anomaly_truths, anomaly_predictions).ravel()
        N_total = anomaly_truths.shape[0]
        N_positives = anomaly_truths.sum()
        N_negatives = N_total - N_positives

        # Sensitivity, hit rate, recall, or true positive rate
        TPR = TP / N_positives
        # Specificity or true negative rate
        TNR = TN / N_negatives
        # Precision or positive predictive value
        PPV = TP / (TP + FP)
        # Negative predictive value
        NPV = TN / (TN + FN)  # TODO debug: "RuntimeWarning: invalid value encountered in longlong_scalars"
        # Fall out or false positive rate
        FPR = FP / N_negatives
        # False negative rate
        FNR = FN / N_positives
        # False discovery rate
        FDR = FP / (FP + TP)
        # accuracy
        ACC = (TP + TN) / (N_positives + N_negatives)
        # balanced accuracy
        balanced_ACC = (TPR + TNR) / 2
        # balanced false rate (BFR)
        BFR = (FPR + FNR) / 2  # could consider making a weighted average here with higher weight on e.g. FPR
        # F1 score
        F1 = 2 * TP / (2 * TP + FP + FN)
        # Area under curve
        AUC = roc_auc_score(anomaly_truths, anomaly_predictions)
        # "mean Average Precision (mAP)"
        AP = average_precision_score(anomaly_truths, anomaly_predictions)
        metrics_dict = {"TP": TP, "TN": TN, "FP": FP, "FN": FN, "TPR": TPR, "TNR": TNR,
                        "precision:": PPV, "NPV": NPV,
                        "FPR": FPR, "FNR": FNR, "FDR": FDR, "accuracy": ACC,
                        "balanced_ACC": balanced_ACC,
                        "BFR": BFR, "F1": F1, "AUC": AUC, "AP": AP}
        df_metrics = pd.DataFrame(metrics_dict, index=[threshold])
        if self.metrics is None:
            self.metrics = df_metrics
            self.metrics.index.name = "threshold"
        else:
            self.metrics = self.metrics.append(df_metrics)

    def print_metrics(self):
        anomaly_preds = list(self.df_samples_latent_and_x_and_preds["AE_anomaly"])
        anomaly_truths = [1 for i in range(len(anomaly_preds))]

        self._compute_performance_metrics(anomaly_truths, anomaly_preds)
        print("\n\n==== Anomaly Detection performance summary ====")
        print("Anomaly detection MSE threshold: {0:1.4f}, train-set quantile: {1}\n".format(self.anomaly_threshold,
                                                                                            self.quantile))
        print("False positive rate: {0:1.2f}".format(self.metrics["FPR"]))
        print("False negative rate: {0:1.2f}".format(self.metrics["FNR"]))
        print("Balanced false rate: {0:1.2f}".format(self.metrics["BFR"]))
        print("AUC score: {0:1.2f}".format(self.metrics["AUC"]))
        print("Average precision: {0:1.2f}".format(self.metrics["AP"]))
        print("F1: {0:1.2f}".format(self.metrics["F1"]))
        print("\nConfusion matrix:\n", confusion_matrix(anomaly_truths, anomaly_preds))
        print("====================================================")

    def _append_anomalies(self, df_nominal, df_anomalies):
        """
        Takes a normal df and an anomaly df and makes an extra column in each, indicating
        if each datapoint is anomalous or nominal. After this, it concatenates them vertically
        and makes sure the timeindex is correct by simply continuing the nominal df's index into
        the anomalous one by incrementing it appropriately

        :param df_nominal: the dataframe with nominal/normal data to have anomalies appended to it
        :type df_nominal: DataFrame
        :param df_anomalies: the dataframe with anomalies to append to df
        :type df_anomalies: DataFrame
        :return: a dataframe containing the nominal data with anomalous data concatted vertically as the last rows
        :rtype: DataFrame
        """

        assert list(df_nominal.columns) == list(df_anomalies.columns)

        df_copy = df_nominal.copy()
        df_anomalies_copy = df_anomalies.copy()

        df_copy = df_copy.assign(synthetic_anomaly=0)
        df_anomalies_copy = df_anomalies_copy.assign(synthetic_anomaly=1)
        N_anomaly = df_anomalies_copy.shape[0]

        if not type(df_nominal.index) == type(df_anomalies.index):
            time_index = pd.to_datetime(df_copy.index)
            time_increment = time_index[-1] - time_index[-2]
            anomaly_index = [time_index[-1] + time_increment]
            for i in range(N_anomaly - 1):
                anomaly_index.append(anomaly_index[i] + time_increment)

        df_anomalies_copy.index = anomaly_index
        df_concatted = pd.concat([df_copy, df_anomalies_copy])
        df_concatted.index = pd.to_datetime(df_concatted.index)
        return df_concatted

    def evaluate(self,
                 df_test_nominal=None,
                 df_test_anomalous=None,
                 anomaly_detector=None,
                 anomaly_mse_quantiles=None,
                 anomaly_thresholds=None):

        df_test = self._append_anomalies(df_test_nominal, df_test_anomalous)
        df_anomaly_truths = df_test["synthetic_anomaly"]
        if anomaly_mse_quantiles is not None:
            anomaly_thresholds = [np.quantile(anomaly_detector.mse_val, q) for q in anomaly_mse_quantiles]

        for threshold in anomaly_thresholds:
            df_anomaly_preds = anomaly_detector.predict(
                df_test.drop("synthetic_anomaly", axis=1),
                anomaly_threshold=threshold
            )
            self._compute_performance_metrics(anomaly_truths=df_anomaly_truths,
                                              anomaly_predictions=df_anomaly_preds["AE_anomaly"],
                                              threshold=threshold)
        print(self.metrics)
        df_ROC = self.metrics[["FPR", "TPR"]]
        plt.plot(df_ROC["FPR"], df_ROC["TPR"])
        plt.show()

    def plot_latent_samples_by_detection(self,
                                         df_vae_latent_space,
                                         df_anomalies_latent,
                                         df_anomalies_predictions,
                                         show=True):

        latent_dim = df_anomalies_latent.shape[1]
        df_anomalies_merged = pd.concat([df_anomalies_latent, df_anomalies_predictions], axis=1)
        df_samples_detected = df_anomalies_merged.loc[df_anomalies_merged["AE_anomaly"] == 1].drop("AE_anomaly", axis=1)
        df_samples_not_detected = df_anomalies_merged.loc[df_anomalies_merged["AE_anomaly"] == 0].drop("AE_anomaly", axis=1)

        df_vae_latent_space_copy = df_vae_latent_space.copy()
        if latent_dim > 2:
            pca = PCA(2)
            df_vae_latent_space_copy = pca.fit_transform(df_vae_latent_space)
            df_samples_detected = pca.transform(df_samples_detected)
            df_samples_not_detected = pca.transform(df_samples_not_detected)

        plt.scatter(df_vae_latent_space_copy.iloc[:, 0], df_vae_latent_space_copy.iloc[:, 1],
                    c="black", s=10, alpha=0.6, label="Nominal")
        plt.scatter(df_samples_detected.iloc[:, 0], df_samples_detected.iloc[:, 1],
                    s=12, c="blue", marker="^", label="Anomaly sample detected")
        plt.scatter(df_samples_not_detected.iloc[:, 0], df_samples_not_detected.iloc[:, 1],
                    s=14, c="red", marker="s", label="Anomaly sample not detected")
        # plot latent space with samples colored by anomaly detection status
        plt.title("Learned normal condition\n vs sampled anomalies")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    pd.set_option('display.max_columns', None)

    # import data
    df = pd.read_csv("train-data-large.csv", index_col="timelocal")
    df = df.dropna()

    N_anomalies = 100
    generator = AnomalyGenerator(12, 2)
    generator.fit(df, epochs=200)
    df_samples_reconstructions, df_samples_latent_space = generator.generate_anomalies(N_anomalies)
    print(df_samples_reconstructions)
    print(df_samples_reconstructions.describe())

    # split data
    x_train, x_test = train_test_split(df, test_size=0.2, shuffle=False)
    x_val, x_test = train_test_split(x_test, test_size=0.5, shuffle=False)

    # scale data
    scaler_train = MinMaxScaler()
    x_train_scaled = scaler_train.fit_transform(x_train)
    x_val_scaled = scaler_train.transform(x_val)
    x_test_scaled = scaler_train.transform(x_test)

    detector = AnomalyDetectorAutoencoder(latent_dim=2, intermediate_dim=12)

    load_model = True
    # load_model = False
    if load_model:
        detector.load()
    else:
        detector.fit(x_train_scaled,
                     x_val_scaled,
                     plot_history=True)
        detector.save()

    threshold = np.quantile(detector.mse_train, 0.99)

    # predict test set and synthesized anomalies
    df_test_anom_preds, x_test_preds = detector.predict(x_test_scaled, threshold)
    df_test_anom_preds.index = x_test.index
    df_samples_anom_preds, x_synth_preds = detector.predict(scaler_train.transform(df_samples_reconstructions),
                                                            threshold)

    # start visualizing the predictions
    visualizer = AnomalyEvaluator()
    # visualizer.collect_visualization_dataframe(df_anomalies_latent,
    #                                            df_anomalies_decoded,
    #                                            df_samples_anom_preds,
    #                                            generator._df_vae_latent_space,
    #                                            x_train,
    #                                            x_test,
    #                                            df_test_anom_preds,
    #                                            threshold)

    visualizer.plot_vae_latent(color_by_columns=["kv_flow", "flush_indicator", "motor_effect"])
    visualizer.plot_vae_latent_with_samples()
    visualizer.plot_vae_latent_samples_by_anomaly_prediction()
    visualizer.plot_anomaly_time_series()
    print("\nEnd of generator script...")
