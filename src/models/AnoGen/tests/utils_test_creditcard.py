import pandas as pd
from sklearn.model_selection import train_test_split
from src.models.AnoGen.anomaly_generator import *
from src.models.AnoGen.utility_functions import *
from sklearn.metrics import precision_recall_curve, average_precision_score


def concat_testset_and_anomalies_rowwise(x_test, x_anomalies):
    return pd.concat([x_test, x_anomalies], axis=0)


def make_binary_labels(N_labels_to_return: int, label: int):
    return [label for i in range(N_labels_to_return)]


def plot_precision_recall_curve(recall_scores, precision_scores):
    plt.plot(recall_scores, precision_scores)
    plt.suptitle("Precision recall curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()