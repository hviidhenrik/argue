"""
Model utilities

"""

import os
from typing import List, Optional, Union

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # silences excessive warning messages from tensorflow

import tensorflow as tf
from tensorflow.keras.models import Model

from argue.config.definitions import *


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
        x = tf.keras.layers.Dense(units, activation=activation)(x)
    return x


class Network:
    def __init__(self, name: str):
        self.name = name
        self.keras_model = None
        self.activation_model = None

    def _init_weight_regularizer(self, l1_weight_penalty, l2_weight_penalty):
        regularizer_combination = (bool(l1_weight_penalty), bool(l2_weight_penalty))
        regularizer_dict = {(False, False): None,
                            (True, False): tf.keras.regularizers.l1(l1_weight_penalty),
                            (False, True): tf.keras.regularizers.l2(l2_weight_penalty),
                            (True, True): tf.keras.regularizers.l1_l2(l1_weight_penalty, l2_weight_penalty)}
        return regularizer_dict[regularizer_combination]

    def build_model(self,
                    input_layer: tf.keras.layers.Layer,
                    output_layer: tf.keras.layers.Layer,
                    units_in_layers: List[int],
                    activation: str = "elu",
                    dropout_frac: Optional[float] = None,
                    keep_output_layer_activations: bool = False,
                    l1_weight_penalty: float = None,
                    l2_weight_penalty: float = None):

        weight_initializer = "he_uniform" if activation == "relu" else "glorot_uniform"
        weight_regularizer = self._init_weight_regularizer(l1_weight_penalty, l2_weight_penalty)

        x = input_layer
        for units in units_in_layers:
            x = tf.keras.layers.Dense(units, activation=activation,
                      kernel_initializer=weight_initializer,
                      kernel_regularizer=weight_regularizer)(x)
            if dropout_frac is not None:
                x = tf.keras.layers.Dropout(dropout_frac)(x)
        outputs = output_layer(x)
        self.keras_model = Model(inputs=input_layer, outputs=outputs, name=self.name)

        # make activation model here
        activation_tensor = extract_activations(self.keras_model, self.name + "_activations",
                                                keep_output_layer=keep_output_layer_activations)
        self.activation_model = Model(inputs=self.keras_model.input, outputs=activation_tensor,
                                      name=self.name + "_activations")
        return self

    def summary(self, model: str = "keras_model"):
        if "keras" in model:
            self.keras_model.summary()
        else:
            self.activation_model.summary()

    def get_hidden_activations(self, x):
        return self.activation_model.predict(x)

    def get_activation_dim(self):
        return self.activation_model.output_shape[1]

    def save(self, path: Union[Path, str] = None):
        self.keras_model.save(path / "keras_model")
        self.activation_model.save(path / "activation_model")

    def load(self, path: Union[Path, str] = None):
        self.keras_model = tf.keras.models.load_model(path / "keras_model", compile=False)
        self.activation_model = tf.keras.models.load_model(path / "activation_model", compile=False)
        return self.keras_model, self.activation_model


class L1CategoricalCrossentropy(tf.keras.losses.Loss):

    def __init__(self):
        super().__init__()
        self.catxe = tf.keras.losses.CategoricalCrossentropy()
        self.l1_loss = tf.keras.losses.MeanAbsoluteError()

    def call(self, y_true, y_pred):
        return self.catxe(y_true, y_pred) + self.l1_loss(y_true, y_pred)


if __name__ == "__main__":
    import tensorflow as tf

    FFN = Network(name="ffn").build_model(input_layer=tf.keras.layers.Input((3,)),
                                          output_layer=tf.keras.layers.Dense(3, "sigmoid"),
                                          units_in_layers=[10, 8, 6],
                                          activation="tanh",
                                          dropout_frac=0.1)
    FFN.summary("keras")
