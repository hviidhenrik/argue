import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model

tf.get_logger().setLevel('ERROR')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from src.utilities.utility_functions import *
from src.data.data_utils import *
from src.config.definitions import *
from src.models.AAE.aae_class import *
import time


class DAD(AdversarialAutoencoder):
    # TODO find out how to integrate the AAE. Inheritance or simply call it within DAD?
    def __init__(self,
                 autoencoder_latent_dimension: int = 2,
                 autoencoder_activation: Union[str, List[str]] = "elu",
                 encoder_hidden_layers: List[int] = None,
                 decoder_hidden_layers: List[int] = None,
                 discrim_hidden_layers: List[int] = None,
                 discrim_activation: Union[str, List[str]] = "elu",
                 anomaly_generator_input_size: int = 7,
                 anomaly_generator_hidden_layers: List[int] = [5, 4],
                 anomaly_generator_activation_function: Union[str, List[str]] = "elu",
                 anomaly_generator_dropout_fraction: float = 0,
                 anomaly_generator_loss_function: str = "mse",
                 anomaly_generator_optimizer: Union[str, tf.keras.optimizers] = "Adam",
                 alarm_hidden_layers: List[int] = [5, 4],
                 alarm_activation_function: Union[str, List[str]] = "elu",
                 alarm_dropout_fraction: float = 0,
                 alarm_loss_function: str = "binary_crossentropy",
                 alarm_optimizer: Union[str, tf.keras.optimizers] = "Adam"
                 ):
        self.autoencoder_latent_dimension = autoencoder_latent_dimension
        self.anomaly_generator_input_size = anomaly_generator_input_size
        self.anomaly_generator_hidden_layers = anomaly_generator_hidden_layers
        self.anomaly_generator_activation_function = anomaly_generator_activation_function
        self.anomaly_generator_dropout_fraction = anomaly_generator_dropout_fraction
        self.anomaly_generator_loss_function = anomaly_generator_loss_function
        self.alarm_hidden_layers = alarm_hidden_layers
        self.alarm_activation_function = alarm_activation_function
        self.alarm_dropout_fraction = alarm_dropout_fraction
        self.alarm_loss_function = alarm_loss_function
        self._anomaly_generator_optimizer = anomaly_generator_optimizer
        self._alarm_optimizer = alarm_optimizer

        super().__init__(autoencoder_latent_dimension, encoder_hidden_layers, decoder_hidden_layers,
                         discrim_hidden_layers, autoencoder_activation, discrim_activation)

    def _build_anomaly_generator(self) -> None:
        generator_input = Input(shape=(self.anomaly_generator_input_size,))
        x = Dense(units=self.anomaly_generator_hidden_layers[0],
                  activation=self.anomaly_generator_activation_function)(generator_input)
        x = Dropout(self.anomaly_generator_dropout_fraction)(x)
        remaining_layers = self.anomaly_generator_hidden_layers[1:]
        for units in remaining_layers:
            x = Dense(units=units, activation=self.anomaly_generator_activation_function)(x)
            x = Dropout(self.anomaly_generator_dropout_fraction)(x)
        latent_code = Dense(self.autoencoder_latent_dimension,
                            activation=self.anomaly_generator_activation_function)(x)
        generator = Model(inputs=generator_input, outputs=latent_code)
        generator.compile(optimizer=self._anomaly_generator_optimizer,
                          loss=self.anomaly_generator_loss_function)
        self.generator = generator

    def _build_alarm(self) -> None:
        # TODO figure out how to get hidden activations from decoder
        alarm_input_size = np.sum(self.decoder_hidden_layers)
        alarm_input = Input(shape=(alarm_input_size,))
        x = Dense(units=self.alarm_hidden_layers[0], activation=self.alarm_activation_function)(alarm_input)
        x = Dropout(self.alarm_dropout_fraction)(x)
        remaining_layers = self.alarm_hidden_layers[1:]
        for units in remaining_layers:
            x = Dense(units=units, activation=self.alarm_activation_function)(x)
            x = Dropout(self.alarm_dropout_fraction)(x)
        anomalous_or_not = Dense(1, activation="sigmoid")(x)
        alarm = Model(inputs=alarm_input, outputs=anomalous_or_not)

        # TODO may have to NOT compile the model as it's not done in aae_class.py's build methods
        alarm.compile(optimizer=self._alarm_optimizer, loss=self.alarm_loss_function)
        self.alarm = alarm

    def fit(self, df_features_scaled: DataFrame, batch_size: int = 128, epochs: int = 10, dropout_fraction: float = 0.2,
            base_learning_rate: float = 0.00025, max_learning_rate: float = 0.0025, verbose: int = 2):
        print("Starting AAE fit!")
        super().fit(df_features_scaled, batch_size, epochs, dropout_fraction, base_learning_rate, max_learning_rate,
                    verbose)
        print("Starting alarm and anomaly generator fit!")
        self._build_anomaly_generator()
        self._build_alarm()


if __name__ == "__main__":
    is_debugging = True
    # is_debugging = False
    pd.set_option('display.max_columns', None)
    filename = get_pump_data_path() / f"data_cwp_pump_10_{get_dataset_purpose_as_str(is_debugging)}.csv"
    df_features = get_local_data(filename)
    df_features = df_features.dropna()
    print(df_features.shape)

    scaler = MinMaxScaler()
    df_features_scaled = scaler.fit_transform(df_features)

    dad = DAD()
    dad.fit(df_features_scaled, epochs=1, batch_size=2024, verbose=3)

    print("Finished")

