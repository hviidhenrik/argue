from dataclasses import dataclass
from pathlib import WindowsPath
import pandas as pd
import numpy as np
from pandas import DataFrame
from typing import List, Union, Optional

import tensorflow as tf
from tensorflow.keras.layers import Dense


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


def partition_by_class():
    pass


def check_alarm_sparsity(y_true, y_pred):
    absolute_error = np.abs(np.sum(y_true, axis=1) - np.sum(y_pred, axis=1))
    return np.mean(absolute_error)


# note authors' original noise generator
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


