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
                 first_hidden_layer_size: int = 12,
                 latent_space_size: int = 2):
        self.feature_cols = None
        self.first_hidden_layer_size = first_hidden_layer_size
        self.latent_space_size = latent_space_size
        self.vae_validation_size = None
        self._x_train = None
        self._x_val = None
        self._scaler = MinMaxScaler()
        self._vae_latent_plot_pca = None
        self.latent_space_stddev = None
        self.neuralnet_activation_function = None
        self.encoder = None
        self.decoder = None
        self.autoencoder = None
        self.latent_space = None
        self.nominal_latent_space_clustering = None
        self.dbscan_epsilon = None
        self.latent_cols = None

    def fit(self,
            df_unscaled,
            vae_validation_size: float = 0.1,
            latent_space_stddev: float = 0.005,
            dbscan_epsilon: float = 0.3,  # 0.3 for 2 dim vae
            activation: str = "elu",
            epochs: int = 200,
            early_stopping: bool = True,
            kl_warmup: int = 30,
            plot_history: bool = False):

        self.df_unscaled_copy = df_unscaled.copy()

        df_train_unscaled, df_val_unscaled = train_test_split(self.df_unscaled_copy,
                                                              test_size=vae_validation_size,
                                                              shuffle=False)
        df_train_scaled = self._scaler.fit_transform(df_train_unscaled)
        df_val_scaled = self._scaler.transform(df_val_unscaled)

        encoder, decoder, vae = fit_VAE(df_train_scaled,
                                        df_val_scaled,
                                        intermediate_dim=self.first_hidden_layer_size,
                                        latent_dim=self.latent_space_size,
                                        batch_size=128,
                                        epochs=epochs,
                                        early_stopping=early_stopping,
                                        kl_warmup=kl_warmup,
                                        latent_stddev=latent_space_stddev,
                                        plot_history=plot_history,
                                        activation=activation)

        self.latent_cols = [f"z{dim_number}" for dim_number in range(self.latent_space_size)]
        df_train_and_val_scaled_together = self._scaler.fit_transform(self.df_unscaled_copy)
        self.latent_space = pd.DataFrame(encoder.predict(df_train_and_val_scaled_together),
                                         columns=self.latent_cols
                                         )

        # fit nominal_latent_space_clustering model to recognize the learned latent space and
        # filter away samples that are
        # overlapping with this learned normal condition
        self.nominal_latent_space_clustering = DBSCANClustering(cols_to_include=self.latent_space.columns.values,
                                                                epsilon=dbscan_epsilon,
                                                                guide_search_for_eps=False,
                                                                plot=False
                                                                )
        self.nominal_latent_space_clustering.fit(self.latent_space)

        self.feature_cols = df_unscaled.columns.values
        self.latent_space_stddev = latent_space_stddev
        self.neuralnet_activation_function = activation
        self.encoder = encoder
        self.decoder = decoder
        self.autoencoder = vae
        self.dbscan_epsilon = dbscan_epsilon
        return self

    def generate_anomalies(self,
                           N_anomalies_to_generate: int = 10,
                           latent_sample_space_minimum: float = -1,
                           latent_sample_space_maximum: float = 1,
                           domain_filter_dict: Dict[str, List[float]] = None):
        """
        Sample anomaly points from the VAE latent space based on the training data.
        """
        def _sample_uniformly_from_latent_space(batch_size_magnitude: int = 10):
            number_of_uniform_samples_to_draw = batch_size_magnitude * N_anomalies_to_generate
            number_of_latent_features = self.latent_space_size
            df_uniform_samples = pd.DataFrame(
                np.random.uniform(low=latent_sample_space_minimum,
                                  high=latent_sample_space_maximum,
                                  size=(number_of_uniform_samples_to_draw, number_of_latent_features)),
                columns=self.latent_cols)
            return df_uniform_samples

        def _concat_dataframes_columnwise(df_1, df_2):
            return pd.concat([df_1, df_2], axis=1)

        def _concat_dataframes_rowwise(df_1, df_2):
            return pd.concat([df_1, df_2], axis=0)

        def _discard_unrealistic_samples(df_latent_and_reconstructions,
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
            nominal_max = self.df_unscaled_copy.max()
            nominal_min = self.df_unscaled_copy.min()
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

        def _identify_samples_outside_normal_condition(df_remaining_realistic_samples):
            df_remaining_realistic_samples_copy = df_remaining_realistic_samples.copy()
            df_outside_nominal = self.nominal_latent_space_clustering.predict(df_remaining_realistic_samples_copy)
            df_remaining_realistic_samples_copy = _concat_dataframes_columnwise(df_remaining_realistic_samples_copy,
                                                                                df_outside_nominal)
            df_remaining_realistic_samples_copy = df_remaining_realistic_samples_copy.loc[
                                                   df_remaining_realistic_samples_copy["anomaly_dbscan"] == 1, :]
            return df_remaining_realistic_samples_copy

        def _get_anomalies_as_decoded_and_latent(df_remaining_realistic_samples):
            df_anomalies_decoded_loop = df_remaining_realistic_samples.iloc[:, self.latent_space_size:].reset_index(
                drop=True)
            df_anomalies_latent_loop = df_remaining_realistic_samples.iloc[:, :self.latent_space_size].reset_index(
                drop=True)
            return df_anomalies_decoded_loop, df_anomalies_latent_loop

        def _discard_abundant_samples(df_anomalies_decoded_final, df_anomalies_latent_final):
            df_anomalies_decoded_final = df_anomalies_decoded_final.iloc[:N_anomalies_to_generate, ]
            df_anomalies_latent_final = df_anomalies_latent_final.iloc[:N_anomalies_to_generate, ]
            return df_anomalies_decoded_final, df_anomalies_latent_final

        samples_accepted = 0
        df_anomalies_decoded_final = pd.DataFrame(columns=self.feature_cols)
        df_anomalies_latent_final = pd.DataFrame(columns=self.latent_cols)
        while samples_accepted < N_anomalies_to_generate:
            df_latent_samples = _sample_uniformly_from_latent_space(batch_size_magnitude=10)
            df_latent_samples_decoded_scaled = self.decoder.predict(df_latent_samples)
            df_latent_samples_decoded_unscaled = pd.DataFrame(
                self._scaler.inverse_transform(df_latent_samples_decoded_scaled),
                columns=self.feature_cols
            )
            df_to_filter = _concat_dataframes_columnwise(df_latent_samples, df_latent_samples_decoded_unscaled)
            df_remaining_realistic_samples = _discard_unrealistic_samples(df_to_filter, domain_filter_dict)
            df_remaining_realistic_samples = _identify_samples_outside_normal_condition(df_remaining_realistic_samples)
            df_remaining_realistic_samples = df_remaining_realistic_samples.drop("anomaly_dbscan", axis=1)
            df_anomalies_decoded_loop, df_anomalies_latent_loop = _get_anomalies_as_decoded_and_latent(df_remaining_realistic_samples)
            df_anomalies_decoded_final = _concat_dataframes_rowwise(df_anomalies_decoded_final, df_anomalies_decoded_loop)
            df_anomalies_latent_final = _concat_dataframes_rowwise(df_anomalies_latent_final, df_anomalies_latent_loop)
            samples_accepted = df_anomalies_decoded_final.shape[0]

        df_anomalies_decoded_final, df_anomalies_latent_final = _discard_abundant_samples(df_anomalies_decoded_final,
                                                                                          df_anomalies_latent_final)
        return df_anomalies_decoded_final, df_anomalies_latent_final

    def get_vae_latent_space(self):
        return self.latent_space

    def plot_vae_latent(self, color_by_columns=None, save=False, show=True, ax=None):
        if color_by_columns is None:
            color_by_columns = [self.df_unscaled_copy.columns[0]]
        elif not isinstance(color_by_columns, list):
            if color_by_columns.lower() == "all":
                color_by_columns = self.df_unscaled_copy.columns.values
            else:
                color_by_columns = [color_by_columns]  # if only a single string is received like "kv_flow"
        df = self.latent_space.copy()
        pca_reduce = False
        title_latent = "AnoGen VAE training latent space"
        if self.latent_space_size > 2:
            pca = PCA(n_components=2).fit(df)
            df = pd.DataFrame(pca.fit_transform(df))
            var_expl = 100 * pca.explained_variance_ratio_.sum()
            title_latent = title_latent + "\nPCA reduced (var explained: {0:4.0f})%".format(var_expl)
            pca_reduce = True
            self._vae_latent_plot_pca = pca

        for coloring_col in color_by_columns:
            plt.scatter(df.iloc[:, 0], df.iloc[:, 1],
                        c=self.df_unscaled_copy[coloring_col], cmap='jet', s=10)
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
