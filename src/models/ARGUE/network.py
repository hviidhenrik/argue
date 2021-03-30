import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # silences excessive warning messages from tensorflow

from tensorflow.keras.models import Model

from src.models.ARGUE.utils import *
from src.config.definitions import *


class Network:
    def __init__(self, name: str):
        self.name = name
        self.keras_model = None
        self.activation_model = None

    def build_model(self,
                    input_layer: tf.keras.layers.Layer,
                    output_layer: tf.keras.layers.Layer,
                    **kwargs):
        inputs = input_layer
        network = network_block(inputs, **kwargs)
        outputs = output_layer(network)
        self.keras_model = Model(inputs=inputs, outputs=outputs, name=self.name)

        # make activation model here
        activation_tensor = extract_activations(self.keras_model, self.name + "_activations")
        self.activation_model = Model(inputs=self.keras_model.input, outputs=activation_tensor)
        return self

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
