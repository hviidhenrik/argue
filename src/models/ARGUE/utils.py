from dataclasses import dataclass
from pathlib import WindowsPath
import pandas as pd
import numpy as np
from pandas import DataFrame
from typing import List, Union, Optional

import tensorflow as tf
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def extract_activations(network: tf.keras.models.Model,
                        target_name: str,
                        keep_output_layer: bool = False) -> tf.keras.layers.Layer:
    """
    Get the activation layers of the defined model

    :param network: model to take the activation layers from, tf.keras.models.Model
    :param target_name: give the target a name, str
    :return: a flattened layer with all hidden activations, tf.keras.layers.Layer
    """

    hidden_dense_layers = []
    for layer in network.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            hidden_dense_layers.append(layer)

    if not keep_output_layer:
        hidden_dense_layers.pop(-1)

    all_activations = []
    for layer_number, layer in enumerate(hidden_dense_layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            flattened_layer = tf.keras.layers.Flatten(name=f"{target_name}_{layer_number}")
            all_activations.append(flattened_layer(layer.output))
        else:
            all_activations.append(layer.output)
    all_activations = tf.keras.layers.Concatenate(name=target_name)(all_activations)
    return all_activations


def vprint(verbose: Union[bool, int], str_to_print: str, **kwargs):
    if verbose:
        print(str_to_print, **kwargs)


def partition_by_quantiles(x, column: str, quantiles: List[float] = None):
    if quantiles is None:
        quantiles = [0, 0.25, 0.5, 0.75, 1.]
    bins = pd.qcut(x[column], quantiles, labels=False)
    x_out = x.copy()
    x_out["partition"] = bins + 1  # add one since the partitions to ARGUE must start in 1
    return x_out


def partition_by_pca_and_clustering(x, n_pca_components: int = 2, n_clusters: int = 2):
    x_pca = PCA(n_pca_components).fit_transform(x)
    clusters = KMeans(n_clusters=n_clusters, n_init=10).fit(x_pca)
    x["partition"] = clusters.labels_ + 1
    return x


def check_alarm_sparsity(y_true, y_pred):
    absolute_error = np.abs(np.sum(y_true, axis=1) - np.sum(y_pred, axis=1))
    return np.mean(absolute_error)


# note: authors' original noise generator
def generate_noise_samples(x: DataFrame, quantiles: Optional[List[float]] = None,
                           n_noise_samples: Optional[int] = None,
                           stdev: float = 1, stdevs_away: float = 3):
    input_dim = pd.DataFrame(x).shape[1]
    # if no noise samples desired, make a df with a single row to keep dimensions intact further on in ARGUE
    if n_noise_samples == 0:
        return pd.DataFrame(np.random.normal(size=(1, input_dim)), columns=x.columns)
    N = x.shape[0] if n_noise_samples is None else n_noise_samples
    noise = np.random.normal(0.5, 1, size=(N, input_dim))
    df_noise = pd.DataFrame((noise), columns=x.columns)
    return df_noise


# note: my modified noise generator
def generate_noise_samples2(x: DataFrame, quantiles: Optional[List[float]] = None,
                            n_noise_samples: Optional[int] = None,
                            stdev: float = 1, stdevs_away: float = 3):
    input_dim = pd.DataFrame(x).shape[1]

    # if no noise samples desired, make a df with a single row to keep dimensions intact further on in ARGUE
    if n_noise_samples == 0:
        return pd.DataFrame(np.random.normal(size=(1, input_dim)), columns=x.columns)

    quantiles = [0.025, 0.975] if quantiles is None else quantiles
    N = x.shape[0] // 2 if n_noise_samples is None else n_noise_samples // 2
    qs = x.quantile(quantiles)
    noise_below = np.random.normal(qs.iloc[0] - stdev * stdevs_away, stdev, size=(N, input_dim))
    noise_above = np.random.normal(qs.iloc[1] + stdev * stdevs_away, stdev, size=(N, input_dim))
    df_noise = pd.DataFrame(np.vstack((noise_below, noise_above)), columns=x.columns)
    return df_noise


# note: another modified noise generator
def generate_noise_samples3(x: DataFrame, n_noise_samples: Optional[int] = None,
                            **kwargs):
    input_dim = pd.DataFrame(x).shape[1]
    n_noise_samples = x.shape[0] // 2 if n_noise_samples is None else n_noise_samples // 2

    # if no noise samples desired, make a df with a single row to keep dimensions intact further on in ARGUE
    if n_noise_samples < 2:
        return pd.DataFrame(np.random.normal(size=(1, input_dim)), columns=x.columns)

    normal_area = (x.min().min(), x.max().max())

    df_noise1 = np.random.normal(size=(n_noise_samples, input_dim))
    df_noise2 = np.random.normal(size=(n_noise_samples, input_dim))
    noise_below = MinMaxScaler(feature_range=(-6, normal_area[0] - 1)).fit_transform(df_noise1)
    noise_above = MinMaxScaler(feature_range=(normal_area[1] + 1, 6)).fit_transform(df_noise2)
    df_noise = pd.DataFrame(np.vstack((noise_below, noise_above)), columns=x.columns)
    return df_noise


def plot_learning_schedule(total_steps: int = None,
                           initial_learning_rate: float = None,
                           decay_rate: float = None,
                           decay_steps: int = None,
                           staircase: bool = False,
                           verbose: bool = False):
    def decayed_learning_rate(step):
        exponent = (step // decay_steps) if staircase else (step / decay_steps)
        return np.power(initial_learning_rate * decay_rate, exponent)

    def get_lr(step):
        return decayed_learning_rate(step) if step > decay_steps else initial_learning_rate

    steps = np.arange(total_steps)
    lr = []
    for step in steps:
        learning_rate = get_lr(step)
        lr.append(learning_rate)
        vprint(verbose, f"Learning rate: {learning_rate:.4f}")

    plt.plot(steps, lr)
    plt.suptitle(f"Learning rate schedule with decay_rate = {decay_rate}")
    plt.xlabel("Step")
    plt.ylabel("Learning rate")


def make_time_elapsed_string(elapsed_time, secs_to_min_threshold: int = 180):
    return f"{elapsed_time:.2f} seconds!" if elapsed_time < secs_to_min_threshold else \
        f"{elapsed_time / 60:.2f} minutes!"
