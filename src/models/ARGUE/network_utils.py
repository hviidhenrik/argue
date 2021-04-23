import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # silences excessive warning messages from tensorflow

from tensorflow.keras.models import Model

from src.models.ARGUE.utils import *
from src.config.definitions import *


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


class Network:
    def __init__(self, name: str):
        self.name = name
        self.keras_model = None
        self.activation_model = None

    def build_model(self,
                    input_layer: tf.keras.layers.Layer,
                    output_layer: tf.keras.layers.Layer,
                    units_in_layers: List[int],
                    activation="elu",
                    dropout_frac: Optional[float] = None,
                    keep_output_layer_activations: bool = False):
        x = input_layer
        for units in units_in_layers:
            x = Dense(units, activation=activation)(x)
            if dropout_frac is not None:
                x = tf.keras.layers.Dropout(dropout_frac)(x)
        outputs = output_layer(x)
        self.keras_model = Model(inputs=input_layer, outputs=outputs, name=self.name)

        # make activation model here
        activation_tensor = extract_activations(self.keras_model, self.name + "_activations",
                                                keep_output_layer=keep_output_layer_activations)
        self.activation_model = Model(inputs=self.keras_model.input, outputs=activation_tensor)
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

    def save(self, path: Union[WindowsPath, str] = None):
        self.keras_model.save(path / "keras_model")
        self.activation_model.save(path / "activation_model")

    def load(self, path: Union[WindowsPath, str] = None):
        self.keras_model = tf.keras.models.load_model(path / "keras_model", compile=False)
        self.activation_model = tf.keras.models.load_model(path / "activation_model", compile=False)
        return self.keras_model, self.activation_model


if __name__ == "__main__":
    import tensorflow as tf
    FFN = Network(name="ffn").build_model(input_layer=tf.keras.layers.Input((3, )),
                                          output_layer=tf.keras.layers.Dense(3, "sigmoid"),
                                          units_in_layers=[10, 8, 6],
                                          activation="tanh",
                                          dropout_frac=0.1)
    FFN.summary("keras")
