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



def scale_and_PCA(df,
                  scaler=None,
                  pca=None,
                  n_components=2,
                  return_objects=False):
    """
    Takes a dataset, scales and applies PCA to it

    :param df:
    :type df:
    :param scaler:
    :type scaler:
    :param pca:
    :type pca:
    :param n_components:
    :type n_components:
    :param return_objects: whether to return _scaler and PCA objects additional to the dataframe
    :type return_objects:
    :return:
    :rtype:
    """
    pc_col_names = ["PC{}".format(number + 1) for number in range(n_components)]

    if scaler is None:
        scaler = MinMaxScaler().fit(df)
    if not scaler:
        df_transformed = df
    else:
        df_transformed = scaler.transform(df)

    if pca is None:
        pca = PCA(n_components=n_components).fit(df_transformed)

    df_scaled_pca = pca.transform(df_transformed)

    if return_objects:
        return df_scaled_pca, scaler, pca
    else:
        return df_scaled_pca


def plot_PCA(df,
             scaler=None,
             pca=None,
             n_components=2,
             show_plot=True):
    """
    Takes a dataset, scales it and applies PCA and plots it. Custom _scaler and PCA objects can be given

    :param df:
    :type df:
    :param scaler:
    :type scaler:
    :param pca:
    :type pca:
    :param n_components:
    :type n_components:
    :param show_plot:
    :type show_plot:
    :return:
    :rtype:
    """
    pc_col_names = ["PC{}".format(number + 1) for number in range(n_components)]

    if scaler is None:
        scaler = MinMaxScaler().fit(df)
    df_scaled = scaler.transform(df)

    if pca is None:
        pca = PCA(n_components=n_components).fit(df_scaled)

    df_scaled_pca = pd.DataFrame(pca.transform(df_scaled), columns=pc_col_names)
    plt.scatter(df_scaled_pca["PC1"], df_scaled_pca["PC2"])
    plt.suptitle("PCA with {0} components\nVar explained: {1:4.2f}%".format(n_components,
                                                                            100 * pca.explained_variance_ratio_.sum()))
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    if show_plot:
        plt.show()


def sample_around_points(df_points, N_samples=10, stddev=0.01):
    samples = []
    for i in range(df_points.shape[0]):
        for j in range(N_samples):
            z1 = normal(loc=df_points.iloc[i, 0], scale=stddev)
            z2 = normal(loc=df_points.iloc[i, 1], scale=stddev)
            samples.append((z1, z2))

    samples = pd.DataFrame(samples, columns=df_points.columns.values)
    return pd.concat([df_points, pd.DataFrame(samples)], axis=0)


def make_grid_2D(n_points, limit):
    n = int(np.floor(np.sqrt(n_points)))
    z0 = np.linspace(limit, -limit, n)
    z1 = np.linspace(limit, -limit, n)
    latent_grid_samples = np.dstack(np.meshgrid(z0, z1)).reshape(-1, 2)
    return pd.DataFrame(latent_grid_samples, columns=["z0", "z1"])


def append_anomalies(df, df_anomalies):
    """
    Takes a normal df and an anomaly df and makes an extra column in each, indicating
    if each datapoint is anomalous or nominal. After this, it concatenates them vertically
    and makes sure the timeindex is correct by simply continuing the nominal df's index into
    the anomalous one by incrementing it appropriately

    :param df: the dataframe with nominal/normal data to have anomalies append to it
    :type df: DataFrame
    :param df_anomalies: the dataframe with anomalies to append to df
    :type df_anomalies: DataFrame
    :return: a dataframe containing the nominal data with anomalous data concatted vertically as the last rows
    :rtype: DataFrame
    """

    assert list(df.columns) == list(df_anomalies.columns)

    df_copy = df.copy()
    df_anomalies_copy = df_anomalies.copy()

    df_copy = df_copy.assign(synthetic_anomaly=0)
    df_anomalies_copy = df_anomalies_copy.assign(synthetic_anomaly=1)
    N_anomaly = df_anomalies_copy.shape[0]

    if not type(df.index) == type(df_anomalies.index):
        time_index = pd.to_datetime(df_copy.index)
        time_increment = time_index[-1] - time_index[-2]
        anomaly_index = [time_index[-1] + time_increment]
        for i in range(N_anomaly - 1):
            anomaly_index.append(anomaly_index[i] + time_increment)

    df_anomalies_copy.index = anomaly_index
    df_concatted = pd.concat([df_copy, df_anomalies_copy])
    df_concatted.index = pd.to_datetime(df_concatted.index)
    return df_concatted

def compute_performance_metrics(y_true, y_predicted):
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
    TN, FP, FN, TP = confusion_matrix(y_true, y_predicted).ravel()
    N_total = len(y_true)
    N_positives = np.sum(y_true)
    N_negatives = N_total - N_positives

    if TP == 0 | TN == 0:
        wrong_prediction_rate = (N_positives - TP + FP + FN + N_negatives - TN) / N_total
        return wrong_prediction_rate
    else:
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
        # AUC = roc_auc_score(y_true, y_predicted)
        # "mean Average Precision (mAP)"
        # AP = average_precision_score(y_true, y_predicted)
        metrics_dict = {"TP": TP, "TN": TN, "FP": FP, "FN": FN, "TPR": TPR, "TNR": TNR,
                        "precision:": PPV, "NPV": NPV,
                        "FPR": FPR, "FNR": FNR, "FDR": FDR, "accuracy": ACC,
                        "balanced_ACC": balanced_ACC,
                        "BFR": BFR, "F1": F1,
                        # "AUC": AUC, "AP": AP
                        }
    # df_metrics = pd.DataFrame(metrics_dict, index=[threshold])
    return metrics_dict

def evaluate_predictions(labels_true, labels_predicted, print_results=True):

    if np.sum(labels_true) == 0 or np.sum(labels_true)/len(labels_true) == 1:
        F1 = f1_score(labels_true, labels_predicted)
        print(f"F1: {F1}")
        return F1
    else:
        metrics = compute_performance_metrics(labels_true, labels_predicted)
        F1 = metrics["F1"]
        FPR = metrics["FPR"]
        TPR = metrics["TPR"]

        if print_results:
            print(f"FPR: {FPR}")
            print(f"TPR: {TPR}")
            print(f"F1-score: {F1}")
            print("Confusion matrix:\n", confusion_matrix(labels_true, labels_predicted))
        return F1, FPR, TPR

