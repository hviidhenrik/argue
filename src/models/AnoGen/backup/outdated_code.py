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
        # cnf_matrix = confusion_matrix(y_true, y_predicted)
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
