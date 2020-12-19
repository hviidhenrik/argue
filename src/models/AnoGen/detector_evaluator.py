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
        # cnf_matrix = confusion_matrix(y_true, y_predicted)
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
            anomaly_thresholds = [np.quantile(anomaly_detector.mse_val_set_actual_vs_predicted, q) for q in anomaly_mse_quantiles]

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
