from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.random import normal
from pdm.models.clustering.dbscan import DBSCANClustering
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src.models.AnoGen.variational_autoencoder import fit_VAE


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
            df_unscaled,
            vae_validation_size: float = 0.1,
            latent_stddev: float = 0.005,
            dbscan_epsilon: float = 0.3,  # 0.3 for 2 dim vae
            activation: str = "elu",
            epochs: int = 200,
            early_stopping: bool = True,
            kl_warmup: int = 30,
            plot_history: bool = False):

        self._df_unscaled_train = df_unscaled.copy()

        # split in train and validation set
        x_train_unscaled, x_val_unscaled = train_test_split(self._df_unscaled_train,
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
                self._scaler.fit_transform(self._df_unscaled_train)),  # refit scaler to whole training set including validation
            columns=self.latent_cols
        )

        # fit clustering model to recognize the learned latent space and
        # filter away samples that are
        # overlapping with this learned normal condition
        self.clustering = DBSCANClustering(cols_to_include=self._df_vae_latent_space.columns.values,
                                           epsilon=dbscan_epsilon,
                                           guide_search_for_eps=False,
                                           plot=False
                                           ).fit(self._df_vae_latent_space)
        self.decoder = decoder
        self.feature_cols = df_unscaled.columns.values
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

            # use the fitted clustering keras_model to filter away samples that are in the part of latent space
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
        nominal_max = self._df_unscaled_train.max()
        nominal_min = self._df_unscaled_train.min()
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
            color_by_columns = [self._df_unscaled_train.columns[0]]
        elif not isinstance(color_by_columns, list):
            if color_by_columns.lower() == "all":
                color_by_columns = self._df_unscaled_train.columns.values
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
                        c=self._df_unscaled_train[coloring_col], cmap='jet', s=10)
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


if __name__ == "__main__":
    pd.set_option('display.max_columns', None)

    # import data
    df = pd.read_csv("..\\..\\data\\SSV_CWP\\train-data-small.csv", index_col="timelocal")
    df = df.dropna()

    N_anomalies = 100
    generator = AnomalyGenerator(12, 2)
    generator.fit(df, epochs=10)
    df_samples_reconstructions, df_samples_latent_space = generator.generate_anomalies(N_anomalies)
    print(df_samples_reconstructions)
    print(df_samples_reconstructions.describe())
