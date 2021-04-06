from dataclasses import dataclass
from pathlib import WindowsPath
import pandas as pd
from typing import List, Union

import tensorflow as tf
from tensorflow.keras.layers import Dense


def extract_activations(network: tf.keras.models.Model,
                        target_name: str,
                        discard_output_layer: bool = True) -> tf.keras.layers.Layer:
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

    if discard_output_layer:
        hidden_dense_layers.pop(-1)

    all_activations = []
    for layer_number, layer in enumerate(hidden_dense_layers):
        flattened_layer = tf.keras.layers.Flatten(name=f"{target_name}_{layer_number}")
        all_activations.append(flattened_layer(layer.output))
    all_activations = tf.keras.layers.Concatenate(name=target_name)(all_activations)
    return all_activations


def network_block(inputs, units_in_layers: List[int], activation="selu"):
    """
    Helper function to create encoder/decoder networks in a clean and easy way.

    :param inputs: input tensor; tf.keras.layers.Input
    :param units_in_layers: a list specifying number of units in each layer; List[int]
    :param activation: activation function to use; str
    :return:
    """
    x = inputs
    for units in units_in_layers:
        x = Dense(units, activation=activation)(x)
    return x


def vprint(verbose: Union[bool, int], str_to_print: str):
    if verbose:
        print(str_to_print)


def partition_in_quantiles(x, column: str, q: List[float] = None):
    if q is None:
        q = [0, 0.25, 0.5, 0.75, 1.]
    bins = pd.qcut(x[column], q, labels=False)
    x_out = x.copy()
    x_out["class"] = bins + 1  # add one since the classes to ARGUE must start in 1
    return x_out


