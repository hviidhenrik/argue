from typing import List, Any, Union, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from pathlib import WindowsPath
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Nadam
from dataclasses import dataclass
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import shuffle
from src.models.autoencoder.base import AutoencoderMixin
from src.models.base.plot_mixin import PlotsMixin, save_or_show_plot
from src.models.ARGUE.utils import *


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


class ARGUE:
    def __init__(self,
                 input_dim: int = 3,
                 number_of_decoders: int = 2,
                 latent_dim: int = 2,
                 ):
        self.input_dim = input_dim
        self.number_of_decoders = number_of_decoders
        self.latent_dim = latent_dim
        self.encoder = None
        self.decoder_dict = {}
        self.autoencoder_dict = {}
        self.alarm = None
        self.gating = None
        self.encoder_activation_dim = None
        self.decoder_activation_dim = None
        self.input_to_alarm_dict = {}
        self.input_to_gating = None

    def build_model(self,
                    encoder_hidden_layers: List[int] = [10, 8, 5],
                    decoders_hidden_layers: List[int] = [5, 8, 10],
                    alarm_hidden_layers: List[int] = [15, 12, 10],
                    gating_hidden_layers: List[int] = [15, 12, 10],
                    encoder_activation: Union[str] = "selu",
                    decoders_activation: Union[str] = "selu",
                    alarm_activation: Union[str] = "selu",
                    gating_activation: Union[str] = "selu", ):
        # set constants
        self.encoder_activation_dim = np.sum(encoder_hidden_layers)
        self.decoder_activation_dim = np.sum(decoders_hidden_layers)

        # build shared encoder
        self.encoder = Network(name="encoder").build_model(Input(shape=(self.input_dim,)),
                                                           Dense(self.latent_dim),
                                                           units_in_layers=encoder_hidden_layers,
                                                           activation=encoder_activation)
        # build alarm network
        self.alarm = Network(name="alarm").build_model(Input(shape=(self.decoder_activation_dim,)),
                                                       Dense(1, "sigmoid"),
                                                       units_in_layers=alarm_hidden_layers,
                                                       activation=alarm_activation)

        # build all decoders/experts and connect them with the shared encoder. Store all in dicts.
        for i in range(1, self.number_of_decoders + 1):
            name = f"decoder_{i}"
            decoder = Network(name=name).build_model(Input(shape=(self.latent_dim,)),
                                                     Dense(self.input_dim),
                                                     units_in_layers=decoders_hidden_layers,
                                                     activation=decoders_activation)
            self.decoder_dict[name] = decoder

            autoencoder = self._make_autoencoder_model(self.decoder_dict[name])
            self.autoencoder_dict[f"autoencoder_{i}"] = autoencoder

            # TODO wrap in function
            inputs = self.encoder.keras_model.input
            x = self.encoder.keras_model(inputs)
            x = decoder.activation_model(x)
            outputs = self.alarm.keras_model(x)
            self.input_to_alarm_dict[f"input_to_alarm_{i}"] = Model(inputs, outputs,
                                                                    name=f"x->encoder->decoder_{i}->alarm->y")

        self.gating = Network(name="gating").build_model(Input(shape=(self.encoder_activation_dim,)),
                                                         Dense(self.number_of_decoders + 1, "softmax"),
                                                         units_in_layers=gating_hidden_layers,
                                                         activation=gating_activation)

        # TODO wrap in function
        inputs = self.encoder.keras_model.input
        x = self.encoder.activation_model(inputs)
        outputs = self.gating.keras_model(x)
        self.input_to_gating = Model(inputs, outputs, name="x->encoder->gating->p")

        return self

    def _make_autoencoder_model(self,
                                decoder: tf.keras.models.Model):
        inputs = self.encoder.keras_model.input
        x = self.encoder.keras_model(inputs)
        outputs = decoder.keras_model(x)
        decoder_number = decoder.name[-1]
        return Model(inputs, outputs, name=f"autoencoder_{decoder_number}")

    def fit(self,
            x_train: Union[DataFrame, np.array],
            partition_labels: Union[DataFrame, List[int]],
            batch_size: int = 128,
            epochs: int = 50,
            number_of_batches: int = 5,
            N_noise_samples: int = None,
            verbose: int = 1):
        """
        Training loop to fit the model.

        :param x: a dataframe or array with
        :param class_labels: a dataframe or list with a priori data point class labels as integers.
        :return:
        """

        labels = list(partition_labels)
        unique_partitions = np.unique(labels)
        vprint(verbose, "Preparing data: slicing into partitions and batches...")
        x_train_copy = x_train.copy()
        x_train_copy = pd.concat([x_train_copy, partition_labels], axis=1)
        x_train_copy.rename(columns={x_train_copy.columns[-1]: "class"})

        N_noise_samples = x_train.shape[0] if N_noise_samples is None else N_noise_samples
        x_train_noise = pd.DataFrame(np.random.normal(loc=0.5, scale=1,
                                                      size=(N_noise_samples, x_train.shape[1])),
                                     columns=x_train.columns)
        x_train_noise["class"] = -1
        x_train_with_noise_and_labels = pd.concat([x_train_copy, x_train_noise]).reset_index(drop=True)
        x_train_with_noise_and_labels = shuffle(x_train_with_noise_and_labels)
        gating_label_vectors = pd.get_dummies(x_train_with_noise_and_labels["class"]).values
        alarm_gating_train_dataset = tf.data.Dataset.from_tensor_slices((x_train_with_noise_and_labels.drop(columns="class"),
                                                                         gating_label_vectors))
        alarm_gating_train_dataset = alarm_gating_train_dataset.batch(batch_size, drop_remainder=True)

        autoencoder_train_dataset_dict = {}
        for partition_number, data_partition in enumerate(unique_partitions):
            train_dataset = x_train_copy[x_train_copy["class"] == data_partition].drop(columns=["class"])
            train_dataset = shuffle(train_dataset)
            partition_batch_size = train_dataset.shape[0] // number_of_batches
            train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset)
            train_dataset = train_dataset.batch(partition_batch_size, drop_remainder=True).prefetch(2)
            autoencoder_train_dataset_dict[f"class_{data_partition}"] = train_dataset
            vprint(verbose, f"Partition {partition_number} batch size: {partition_batch_size}, "
                            f"number of batches: {number_of_batches}")

        # NOTE: using one optimizer and loss function for all decoders for now. Could try one for each...
        ae_optimizer = tf.keras.optimizers.RMSprop()
        ae_loss = tf.losses.MeanSquaredError()
        ae_metric = tf.metrics.MeanAbsoluteError()

        # first train encoder and decoders
        vprint(verbose, "\n=== Phase 1: training autoencoder pairs ===")
        for epoch in range(epochs):
            vprint(verbose, f"\n>> Epoch {epoch}")
            for name, model in self.autoencoder_dict.items():
                epoch_loss = []
                epoch_metric = []
                vprint(verbose > 1, f"== Model: {name}, training steps:")
                # for each model, iterate over all its batches from its own dataset and update weights
                # TODO in the future, if this doesnt work well, training should alternate between models,
                #  1 batch for each at a time

                partition = name[-1]
                for step, x_batch_train in enumerate(autoencoder_train_dataset_dict[f"class_{partition}"]):
                    with tf.GradientTape() as tape:
                        predictions = model(x_batch_train, training=True)
                        loss_value = ae_loss(x_batch_train, predictions)

                    gradients = tape.gradient(loss_value, model.trainable_weights)
                    ae_optimizer.apply_gradients(zip(gradients, model.trainable_weights))
                    ae_metric.update_state(x_batch_train, predictions)
                    error_metric = ae_metric.result()
                    epoch_metric.append(error_metric)
                    ae_metric.reset_states()

                    loss_value = float(loss_value)
                    epoch_loss.append(loss_value)
                    if step % 1 == 0 and verbose > 1:
                        print(f"Batch {step} training loss: {loss_value:.4f}, "
                              f"MAE: {float(error_metric):.4f}")

                vprint(verbose, f"Model {name[-1]} epoch loss: {np.mean(epoch_loss):.4f}, "
                                f"MAE: {np.mean(epoch_metric):.4f}")

        # set encoder and decoders to be non-trainable
        # TODO wrap in function
        self.encoder.keras_model.trainable = False
        for name, network_object in self.decoder_dict.items():
            network_object.keras_model.trainable = False

        # train alarm network
        alarm_optimizer = tf.keras.optimizers.RMSprop()
        alarm_loss = tf.losses.BinaryCrossentropy()
        alarm_metric = tf.metrics.BinaryAccuracy()

        vprint(verbose, "\n=== Phase 2: training alarm network ===")
        for epoch in range(epochs):
            vprint(verbose, f"\n>> Epoch {epoch}")
            epoch_loss = []
            epoch_metric = []
            for step, (x_batch_train, true_gating) in enumerate(alarm_gating_train_dataset):
                vprint(step % 20 == 0 and verbose > 1, f"\nStep: {step}")
                for name, model in self.input_to_alarm_dict.items():
                    with tf.GradientTape(persistent=True) as tape:
                        predicted_alarm = model(x_batch_train, training=True)
                        true_alarm = (1 - true_gating.numpy())[:, int(name[-1])].reshape((-1, 1))
                        loss_value = alarm_loss(true_alarm, predicted_alarm)
                        vprint(step % 20 == 0 and verbose > 1,
                               f"Alarm model {name} batch loss: {float(loss_value)}")

                    gradients = tape.gradient(loss_value, self.alarm.keras_model.trainable_weights)
                    alarm_optimizer.apply_gradients(zip(gradients, self.alarm.keras_model.trainable_weights))
                    alarm_metric.update_state(true_alarm, predicted_alarm)
                    error_metric = alarm_metric.result()
                    epoch_metric.append(error_metric)
                    alarm_metric.reset_states()
                epoch_loss.append(float(loss_value))

                if step % 40 == 0 and verbose > 1:
                    print(f"Batch {step} training loss: {float(loss_value):.4f}, ")

            vprint(verbose, f"Alarm epoch loss: {np.mean(epoch_loss):.4f}, "
                            f"Binary accuracy: {np.mean(epoch_metric):.4f}")

        # set alarm to be non-trainable
        # TODO wrap in function
        self.alarm.keras_model.trainable = False

        # train gating network
        gating_optimizer = tf.keras.optimizers.RMSprop()
        gating_loss = tf.losses.CategoricalCrossentropy()
        gating_metric = tf.metrics.CategoricalAccuracy()

        # TODO this will be faster if done inside the same training loop as the alarm model,
        #  but kept separate for easier implementation and getting the details right
        vprint(verbose, "\n=== Phase 3: training gating network ===")
        for epoch in range(epochs):
            vprint(verbose, f"\n>> Epoch {epoch}")
            for step, (x_batch_train, true_gating) in enumerate(alarm_gating_train_dataset):
                epoch_loss = []
                epoch_metric = []
                with tf.GradientTape() as tape:
                    model = self.input_to_gating
                    predicted_gating = model(x_batch_train, training=True)
                    loss_value = gating_loss(true_gating, predicted_gating)
                    epoch_loss.append(float(loss_value))
                    loss_value += loss_value

                gradients = tape.gradient(loss_value, self.gating.keras_model.trainable_weights)
                gating_optimizer.apply_gradients(zip(gradients, self.gating.keras_model.trainable_weights))
                gating_metric.update_state(true_gating, predicted_gating)
                error_metric = gating_metric.result()
                epoch_metric.append(error_metric)
                gating_metric.reset_states()

                if step % 40 == 0 and verbose > 1:
                    print(f"Batch {step} training loss: {float(loss_value):.4f}, ")

            vprint(verbose, f"Gating epoch loss: {np.mean(epoch_loss):.4f}, "
                            f"Categorical accuracy: {np.mean(epoch_metric):.4f}")

        vprint(verbose, "\nModel fitted!")

    def predict(self, x):
        gating_vector = self.input_to_gating.predict(x).reshape((-1, self.number_of_decoders + 1))
        alarm_vector = []

        # get alarm scores from each decoder model and some reshaping afterwards
        for name, model in self.input_to_alarm_dict.items():
            alarm_vector.append(model.predict(x))
        alarm_vector = np.array(alarm_vector).reshape((self.number_of_decoders, -1))

        # stack the virtual decision vector on top of the model alarm verdicts vector
        alarm_vector = np.vstack([np.ones((1, x.shape[0])), alarm_vector]).transpose()

        # compute final weighted average anomaly score
        predictions = np.multiply(gating_vector, alarm_vector).sum(axis=1)
        return predictions


@dataclass
class SubmodelContainer:
    keras_model: tf.keras.models.Model
    loss_function: tf.keras.losses.Loss
    optimizer: tf.keras.optimizers.Optimizer


if __name__ == "__main__":
    # make some data
    N_1 = 5000
    N_2 = 5000
    N_3 = 5000
    df = pd.concat([
        pd.DataFrame({"x1": 4 + np.sin(np.linspace(0, 10, N_1) + np.random.normal(0, 0.1, N_1)),
                      "x2": 4 + np.cos(np.linspace(0, 10, N_1) + np.random.normal(0, 0.1, N_1)),
                      "x3": 4 + np.cos(3.14 + np.linspace(0, 10, N_1) + np.random.normal(0, 0.1, N_1)),
                      "class": 1
                      }),
        pd.DataFrame({"x1": 500 + np.sin(np.linspace(0, 10, N_2) + np.random.normal(0, 0.1, N_2)),
                      "x2": 500 + np.cos(np.linspace(0, 10, N_2) + np.random.normal(0, 0.1, N_2)),
                      "x3": 500 + np.cos(3.14 + np.linspace(0, 10, N_2) + np.random.normal(0, 0.1, N_2)),
                      "class": 2
                      }),
        pd.DataFrame({"x1": -100 - 2 * np.linspace(0, 10, N_3) + np.random.normal(0, 0.1, N_3),
                      "x2": -100 - 3 * np.linspace(0, 10, N_3) + np.random.normal(0, 0.1, N_3),
                      "x3": -100 - 1 * np.linspace(0, 10, N_3) + np.random.normal(0, 0.1, N_3),
                      "class": 3
                      })
    ]
    ).reset_index(drop=True)
    # df.plot(subplots=True)
    # plt.show()

    df[["x1", "x2", "x3"]] = StandardScaler().fit_transform(df[["x1", "x2", "x3"]])

    model = ARGUE(number_of_decoders=3).build_model()
    model.fit(df.drop(columns=["class"]), df["class"], epochs=1, number_of_batches=8, batch_size=256,
              verbose=1, N_noise_samples=N_1)

    indices = [0, N_2 - 1, N_2 + N_3 - 1]
    x_new = df[["x1", "x2", "x3"]].iloc[indices]
    anomaly = pd.DataFrame({"x1": [100000 for _ in range(6)],
                            "x2": [100000 for _ in range(6)],
                            "x3": [100000 for _ in range(6)]})
    x_new = pd.concat([x_new, anomaly]).reset_index(drop=True)
    # print(x_new)
    preds = model.predict(x_new)
    print(preds)
