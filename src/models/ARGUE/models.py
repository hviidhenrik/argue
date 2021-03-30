import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # silences excessive warning messages from tensorflow

import pickle
import numpy as np
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm
from typing import Dict, List, Union
from sklearn.utils import shuffle
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.python.keras.engine.functional import Functional

from src.models.ARGUE.utils import *
from src.models.ARGUE.network import Network
from src.config.definitions import *


# TODO:
#  Required:
#  - make training loops use validation data
#  - make plotting features
#    - learning curves
#    - times series plots where alarm percentage is displayed
#    - if weighted autoencoder reconstructions are implemented, make predicted vs observed plots
#   - a clustering method could be standard partitioning method, if no class vector is given
#   - model shouldn't care if class labels start at 0 or 1 or whatever
#  Nice to have:
#  - make data handling more clean (maybe make a class for handling this)
#  - make reconstructions from decoders available using the weighted average (using the gating vector)
#  - make build_model able to take Network class to specify submodels more flexibly
#  - fit method could have argument to specify model optimizers, loss and metrics (maybe just optim)
#  - class ARGUEPrinter that takes an ARGUE obj and prints nicely readable output from it


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
        self.verbose = None

    def _connect_autoencoder_pair(self, name):
        decoder = self.decoder_dict[name]
        decoder_number = decoder.name[-1]
        inputs = self.encoder.keras_model.input
        x = self.encoder.keras_model(inputs)
        outputs = decoder.keras_model(x)
        return Model(inputs, outputs, name=f"autoencoder_{decoder_number}")

    def _connect_alarm_pair(self, name):
        decoder = self.decoder_dict[name]
        decoder_number = decoder.name[-1]
        inputs = self.encoder.keras_model.input
        x = self.encoder.keras_model(inputs)
        x = decoder.activation_model(x)
        outputs = self.alarm.keras_model(x)
        return Model(inputs, outputs, name=f"input_to_alarm_{decoder_number}")

    def _connect_gating(self):
        inputs = self.encoder.keras_model.input
        x = self.encoder.activation_model(inputs)
        outputs = self.gating.keras_model(x)
        return Model(inputs, outputs, name="input_to_gating")

    def _make_non_trainable(self, submodel: str = None):
        if submodel == "autoencoders":
            self.encoder.keras_model.trainable = False
            for name, network_object in self.decoder_dict.items():
                network_object.keras_model.trainable = False
        if submodel == "alarm":
            self.alarm.keras_model.trainable = False

    def build_model(self,
                    encoder_hidden_layers: List[int] = [10, 8, 5],
                    decoders_hidden_layers: List[int] = [5, 8, 10],
                    alarm_hidden_layers: List[int] = [15, 12, 10],
                    gating_hidden_layers: List[int] = [15, 12, 10],
                    encoder_activation: Union[str] = "selu",
                    decoders_activation: Union[str] = "selu",
                    alarm_activation: Union[str] = "selu",
                    gating_activation: Union[str] = "selu",
                    all_activations: str = None):
        if all_activations is not None:
            for activation in [encoder_activation, decoders_activation,
                               alarm_activation, gating_activation]:
                activation = all_activations
        # set constants
        self.encoder_activation_dim = int(np.sum(encoder_hidden_layers))
        self.decoder_activation_dim = int(np.sum(decoders_hidden_layers))

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

        self.gating = Network(name="gating").build_model(Input(shape=(self.encoder_activation_dim,)),
                                                         Dense(self.number_of_decoders + 1, "softmax"),
                                                         units_in_layers=gating_hidden_layers,
                                                         activation=gating_activation)

        # build all decoders/experts and connect them with the shared encoder. Store all in dicts.
        for i in range(1, self.number_of_decoders + 1):
            decoder_name = f"decoder_{i}"
            self.decoder_dict[decoder_name] = \
                Network(name=decoder_name).build_model(Input(shape=(self.latent_dim,)),
                                                       Dense(self.input_dim),
                                                       units_in_layers=decoders_hidden_layers,
                                                       activation=decoders_activation)

            # connect common encoder with each decoder and store in a dictionary
            self.autoencoder_dict[f"autoencoder_{i}"] = self._connect_autoencoder_pair(decoder_name)

            # connect encoder with alarm model through each decoder/expert network
            self.input_to_alarm_dict[f"input_to_alarm_{i}"] = self._connect_alarm_pair(decoder_name)

        self.input_to_gating = self._connect_gating()
        return self

    def fit(self,
            x_train: Union[DataFrame, np.array],
            partition_labels: Union[DataFrame, List[int]],
            batch_size: int = 128,
            epochs: int = 50,
            number_of_batches: int = 5,
            n_noise_samples: int = None,
            verbose: int = 1,
            optimizer: Union[tf.keras.optimizers.Optimizer, str] = "adam"):

        self.verbose = verbose

        # form initial training data making sure labels and classes are right
        unique_partitions = np.unique(list(partition_labels))
        vprint(verbose, "Preparing data: slicing into partitions and batches...")
        x_train_copy = x_train.copy()
        x_train_copy = pd.concat([x_train_copy, partition_labels], axis=1)
        x_train_copy.rename(columns={x_train_copy.columns[-1]: "class"})

        # make gaussian noise samples so the optimization doesn't only see "healthy" data
        # and hence just learns to always predict healthy, i.e. P(healthy) = certain
        n_noise_samples = x_train.shape[0] if n_noise_samples is None else n_noise_samples
        x_train_noise = pd.DataFrame(np.random.normal(loc=0.5, scale=1,
                                                      size=(n_noise_samples, x_train.shape[1])),
                                     columns=x_train.columns)
        x_train_noise["class"] = -1
        x_train_with_noise_and_labels = pd.concat([x_train_copy, x_train_noise]).reset_index(drop=True)
        x_train_with_noise_and_labels = shuffle(x_train_with_noise_and_labels)

        # make training set for the alarm and gating networks
        gating_label_vectors = pd.get_dummies(x_train_with_noise_and_labels["class"]).values
        alarm_gating_train_dataset = tf.data.Dataset.from_tensor_slices(
            (x_train_with_noise_and_labels.drop(columns="class"),
             gating_label_vectors))
        alarm_gating_train_dataset = alarm_gating_train_dataset.shuffle(1024).batch(batch_size,
                                                                                    drop_remainder=True)

        # make training set for the autoencoder pairs
        autoencoder_train_dataset_dict = {}
        for partition_number, data_partition in enumerate(unique_partitions):
            train_dataset = x_train_copy[x_train_copy["class"] == data_partition].drop(columns=["class"])
            train_dataset = shuffle(train_dataset)
            partition_batch_size = train_dataset.shape[0] // number_of_batches
            train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset)
            train_dataset = train_dataset.shuffle(1024).batch(partition_batch_size,
                                                              drop_remainder=True).prefetch(2)
            autoencoder_train_dataset_dict[f"class_{data_partition}"] = train_dataset
            vprint(verbose, f"Autoencoder data partition {partition_number} batch size: "
                            f"{partition_batch_size}, number of batches: {number_of_batches}")

        # NOTE: using one optimizer and loss function for all decoders for now. Could try one for each...
        # first train encoder and decoders
        vprint(verbose, "\n\n=== Phase 1: training autoencoder pairs ===")
        ae_optimizer = tf.keras.optimizers.get(optimizer)
        ae_loss = tf.losses.MeanSquaredError()
        ae_metric = tf.metrics.MeanAbsoluteError()
        for epoch in range(epochs):
            vprint(verbose, f"\n>> Epoch {epoch}")
            total_model_loss = []
            total_model_metric = []
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

                vprint(verbose > 1, f"Model {name[-1]} epoch loss: {np.mean(epoch_loss):.4f}, "
                                    f"MAE: {np.mean(epoch_metric):.4f}")
                total_model_loss.append(np.mean(epoch_loss))
                total_model_metric.append(np.mean(epoch_metric))
            vprint(verbose, f"--- Average epoch loss: {np.mean(total_model_loss):.4f}, "
                            f"average MAE: {np.mean(total_model_metric):.4f}")
        self._make_non_trainable("autoencoders")

        # train alarm network
        vprint(verbose, "\n\n=== Phase 2: training alarm network ===")
        alarm_optimizer = tf.keras.optimizers.get(optimizer)
        alarm_loss = tf.losses.BinaryCrossentropy()
        alarm_metric = tf.metrics.BinaryAccuracy()
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

        self._make_non_trainable("alarm")

        # train gating network
        vprint(verbose, "\n\n=== Phase 3: training gating network ===")
        gating_optimizer = tf.keras.optimizers.get(optimizer)
        gating_loss = tf.losses.CategoricalCrossentropy()
        gating_metric = tf.metrics.CategoricalAccuracy()
        # TODO this will be faster if done inside the same training loop as the alarm model,
        #  but kept separate for easier implementation and getting the details right
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

        vprint(verbose, "\n----------- Model fitted!\n\n")

    def predict(self, x):
        gating_vector = self.predict_gating_weights(x)
        alarm_vector = self.predict_alarm_probabilities(x).transpose()

        # stack the virtual decision vector on top of the alarm verdicts vector
        alarm_vector = np.vstack([np.ones((1, x.shape[0])), alarm_vector]).transpose()

        # compute final weighted average anomaly score
        predictions = np.multiply(gating_vector, alarm_vector).sum(axis=1)
        return predictions

    def predict_gating_weights(self, x):
        return self.input_to_gating.predict(x).reshape((-1, self.number_of_decoders + 1))

    def predict_alarm_probabilities(self, x):
        alarm_vector = [model.predict(x) for _, model in self.input_to_alarm_dict.items()]
        return np.array(alarm_vector).reshape((self.number_of_decoders, -1)).transpose()

    def save(self, path: Union[WindowsPath, str] = None, model_name: str = None):
        def _save_models_in_dict(model_dict: Dict):
            for name, model in model_dict.items():
                model.save(path / name)

        vprint(self.verbose, "Saving model...\n")
        model_name = model_name if model_name else "argue"
        path = get_model_archive_path() / model_name if not path else path

        # iterate over all the different item types in the self dictionary to be saved
        non_model_attributes_dict = {}
        with tqdm(total=len(vars(self))) as pbar:
            for name, attribute in vars(self).items():
                if isinstance(attribute, Network):
                    attribute.save(path / attribute.name)
                elif isinstance(attribute, dict):
                    _save_models_in_dict(attribute)
                elif isinstance(attribute, Functional):
                    attribute.save(path / name)
                else:
                    non_model_attributes_dict[name] = attribute
                pbar.update(1)

        with open(path / "non_model_attributes.pkl", "wb") as file:
            pickle.dump(non_model_attributes_dict, file)

        vprint(self.verbose, f"... Model saved succesfully in {path}")

    def load(self, path: Union[WindowsPath, str] = None, model_name: str = None):
        print("Loading model...")
        model_name = model_name if model_name else "argue"
        path = get_model_archive_path() / model_name if not path else path

        # finally, load the dictionary storing the builtin/simple types, e.g. ints
        with open(path / "non_model_attributes.pkl", "rb") as file:
            non_model_attributes_dict = pickle.load(file)
        for name, attribute in non_model_attributes_dict.items():
            vars(self)[name] = attribute

        # an untrained model needs to be built before we can start loading it
        self.build_model()

        # iterate over all the different item types to be loaded into the untrained model
        with tqdm(total=len(vars(self))) as pbar:
            for name, attribute in vars(self).items():
                if isinstance(attribute, Network):
                    attribute.load(path / name)
                elif isinstance(attribute, Functional):
                    vars(self)[name] = tf.keras.models.load_model(path / name, compile=False)
                elif isinstance(attribute, dict):
                    for item_name, item_in_dict in attribute.items():
                        if isinstance(item_in_dict, Network):
                            item_in_dict.load(path / item_in_dict.name)
                        elif isinstance(item_in_dict, Functional):
                            vars(self)[name][item_name] = tf.keras.models.load_model(path / item_name, compile=False)
                pbar.update(1)

        vprint(self.verbose, "... Model loaded and ready!")

        return self
