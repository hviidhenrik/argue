import os
import pickle
from typing import Dict, Optional, Collection

import numpy as np
import pandas as pd
from tqdm import tqdm
from pandas import DataFrame
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.python.keras.engine.functional import Functional
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.special import softmax

from src.models.ARGUE.utils import *
from src.models.ARGUE.network_utils import Network, OnesLayer
from src.config.definitions import *

plt.style.use('seaborn')


# TODO:
#  Required:
#  - make plotting features
#     - learning curves
#  - make AUC evaluation
#  - make one autoencoder model with multiple outputs, like the alarm model
#  Nice to have:
#  - a clustering method could be standard partitioning method, if no class vector is given
#  - make data handling more clean (maybe make a class for handling this)
#  - make build_model able to take Network class to specify submodels more flexibly
#  - class ARGUEPrinter that takes an ARGUE obj and prints nicely readable output from it
#  - more realistic anomalies for the noise counter examples
#  - restore best weights
#  - early stopping
#  Experimental:
#  - try separate optimizers and loss for the autoencoder pairs
#  - could the raw alarm probabilities be used without the gating if we simply take the minimum probability over all
#    the models for each datapoint?
#  - could data be sliced vertically instead of horizontally? So each decoder is responsible for a
#    predetermined set of variables instead of rows? Could also be used to model several pumps at the same time
#  - look into variable importance / fault contribution analysis
#  - look into decorrelating the variables in the latent space
#

class ARGUE:
    def __init__(self,
                 input_dim: int = 3,
                 number_of_decoders: int = 2,
                 latent_dim: int = 2,
                 verbose: int = 1,
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
        self.input_to_alarm = None
        self.input_to_gating = None
        self.history = None
        self.verbose = verbose

    def _connect_autoencoder_pair(self, name):
        decoder = self.decoder_dict[name]
        decoder_number = decoder.name[-1]
        inputs = self.encoder.keras_model.input
        x = self.encoder.keras_model(inputs)
        outputs = decoder.keras_model(x)
        return Model(inputs, outputs, name=f"autoencoder_{decoder_number}")

    def _connect_alarm_pair(self, name, use_encoder: bool = False):
        decoder = self.decoder_dict[name]
        decoder_number = decoder.name[-1]
        inputs = self.encoder.keras_model.input
        x = self.encoder.keras_model(inputs)
        x = decoder.activation_model(x)
        if use_encoder:
            y = self.encoder.activation_model(inputs)
            x = tf.keras.layers.concatenate([x, y])
        outputs = self.alarm.keras_model(x)
        return Model(inputs, outputs, name=f"input_to_alarm_{decoder_number}")

    def _connect_gating(self):
        inputs = self.encoder.activation_model.input
        x = self.encoder.activation_model(inputs)
        outputs = self.gating.keras_model(x)
        return Model(inputs, outputs, name="input_to_gating")

    def _make_non_trainable(self, submodel: str = None):
        if submodel == "autoencoders":
            self.encoder.keras_model.trainable = False
            for name, network_object in self.decoder_dict.items():
                network_object.keras_model.trainable = False
        elif submodel == "alarm":
            self.alarm.keras_model.trainable = False
        elif submodel == "gating":
            self.gating.keras_model.trainable = False
        else:
            pass

    @staticmethod
    def _autoencoder_step(x, model, loss, optimizer, metric, training: bool):
        with tf.GradientTape() as tape:
            predictions = model(x, training=training)
            loss_value = loss(x, predictions)
        if training:
            gradients = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        metric.update_state(x, predictions)
        return loss_value

    @staticmethod
    @tf.function
    def _ae_validation_step(x_batch_val, ae_model, ae_loss_fn, ae_metric_fn):
        val_predictions = ae_model(x_batch_val, training=False)
        loss_value = ae_loss_fn(x_batch_val, val_predictions)
        ae_metric_fn.update_state(x_batch_val, val_predictions)
        return loss_value

    @tf.function
    def _alarm_gating_validation_step(self, x_batch_val, true_gating, alarm_loss, gating_loss, final_loss,
                                      alarm_metric, gating_metric, final_metric):
        true_alarm = 1 - true_gating[:, 1:]
        true_final = tf.reduce_min(true_alarm, axis=1)
        predicted_alarm = self.input_to_alarm(x_batch_val, training=False)
        if type(predicted_alarm) == list:
            predicted_alarm = tf.stack(predicted_alarm, axis=2)
            predicted_alarm = tf.reshape(predicted_alarm, shape=(-1, self.number_of_decoders))
        alarm_and_ones = tf.concat([tf.ones((predicted_alarm.shape[0], 1)), predicted_alarm], axis=1)
        predicted_gating = self.input_to_gating(x_batch_val, training=False)
        predicted_prod = tf.multiply(predicted_gating, alarm_and_ones)
        predicted_final = tf.reduce_sum(predicted_prod, axis=1)

        alarm_loss_value = alarm_loss(true_alarm, predicted_alarm)
        gating_loss_value = gating_loss(true_gating, predicted_gating)
        final_loss_value = final_loss(true_final, predicted_final)
        alarm_metric.update_state(true_alarm, predicted_alarm)
        gating_metric.update_state(true_gating, predicted_gating)
        final_metric.update_state(true_final, predicted_final)
        total_loss = alarm_loss_value + gating_loss_value + final_loss_value
        return total_loss, alarm_loss_value, gating_loss_value, final_loss_value

    @staticmethod
    def _init_loss_functions():
        ae_loss = tf.losses.BinaryCrossentropy()
        alarm_loss = tf.losses.BinaryCrossentropy()
        gating_loss = tf.losses.CategoricalCrossentropy()
        final_loss = tf.losses.BinaryCrossentropy()
        return ae_loss, alarm_loss, gating_loss, final_loss

    @staticmethod
    def _init_metric_functions():
        ae_metric = tf.metrics.MeanAbsoluteError()
        alarm_metric = tf.metrics.MeanAbsoluteError()
        gating_metric = tf.metrics.MeanAbsoluteError()
        final_metric = tf.metrics.MeanAbsoluteError()
        return ae_metric, alarm_metric, gating_metric, final_metric

    def build_model(self,
                    encoder_hidden_layers: List[int] = [10, 8, 5],
                    decoders_hidden_layers: List[int] = [5, 8, 10],
                    alarm_hidden_layers: List[int] = [15, 12, 10],
                    gating_hidden_layers: List[int] = [15, 12, 10],
                    encoder_activation: str = "tanh",
                    decoders_activation: str = "tanh",
                    alarm_activation: str = "tanh",
                    gating_activation: str = "tanh",
                    all_activations: Optional[str] = None,
                    use_encoder_activations_in_alarm: bool = False):

        # if all_activations is specified, the same activation function is used in all hidden layers
        if all_activations is not None:
            encoder_activation = all_activations
            decoders_activation = all_activations
            alarm_activation: all_activations
            gating_activation: all_activations

        # set constants
        self.encoder_activation_dim = int(np.sum(encoder_hidden_layers))
        self.decoder_activation_dim = int(np.sum(decoders_hidden_layers))

        # build shared encoder
        self.encoder = Network(name="encoder").build_model(input_layer=Input(shape=(self.input_dim,)),
                                                           output_layer=Dense(self.latent_dim, encoder_activation),
                                                           units_in_layers=encoder_hidden_layers,
                                                           activation=encoder_activation)
        # build alarm network
        alarm_input_dim = self.decoder_activation_dim
        if use_encoder_activations_in_alarm:
            alarm_input_dim += self.encoder_activation_dim
        self.alarm = Network(name="alarm").build_model(input_layer=Input(shape=(alarm_input_dim,)),
                                                       output_layer=Dense(1, "sigmoid"),
                                                       units_in_layers=alarm_hidden_layers,
                                                       activation=alarm_activation)

        self.gating = Network(name="gating").build_model(input_layer=Input(shape=(self.encoder_activation_dim,)),
                                                         output_layer=Dense(self.number_of_decoders + 1, "softmax"),
                                                         units_in_layers=gating_hidden_layers,
                                                         activation=gating_activation)

        # build all decoders/experts and connect them with the shared encoder. Store all in dicts.
        # TODO: convert this to one model with multiple outputs like the alarm model
        alarm_outputs = []
        for i in range(1, self.number_of_decoders + 1):
            decoder_name = f"decoder_{i}"
            self.decoder_dict[decoder_name] = \
                Network(name=decoder_name).build_model(input_layer=Input(shape=(self.latent_dim,)),
                                                       output_layer=Dense(self.input_dim, "sigmoid"),
                                                       units_in_layers=decoders_hidden_layers,
                                                       activation=decoders_activation)

            # connect common encoder with each decoder and store in a dictionary
            self.autoencoder_dict[f"autoencoder_{i}"] = self._connect_autoencoder_pair(decoder_name)

            # connect encoder with alarm model through each decoder/expert network
            alarm_outputs.append(self._connect_alarm_pair(decoder_name,
                                                          use_encoder_activations_in_alarm).output)

        self.input_to_alarm = Model(inputs=self.encoder.keras_model.input,
                                    outputs=alarm_outputs)
        self.input_to_gating = self._connect_gating()

        # predicted_alarm = tf.stack(alarm_model(x_batch_train, training=True), axis=2)
        # predicted_alarm = tf.reshape(predicted_alarm, shape=(-1, N_decoders))
        # alarm_and_ones = tf.concat([tf.ones((predicted_alarm.shape[0], 1)), predicted_alarm], axis=1)
        # predicted_gating = gating_model(x_batch_train, training=True)
        # predicted_prod = tf.multiply(predicted_gating, alarm_and_ones)
        # predicted_final = tf.reduce_sum(predicted_prod, axis=1)

        # def fun():
        #     x = tf.ones(shape=(1, ))
        #
        #
        # inputs = self.encoder.keras_model.input
        # alarm = self.input_to_alarm(inputs)
        # alarm = tf.keras.layers.concatenate(alarm)
        # # alarm = tf.keras.layers.Reshape(target_shape=(None, self.number_of_decoders))
        # # ones = tf.keras.layers.Lambda(lambda x: fun(), (None, 1))(alarm)
        # # ones = OnesLayer()(alarm)
        # alarm_and_ones = tf.keras.layers.concatenate([input_constant, alarm])
        # gating = self.input_to_gating(inputs)
        # prod = tf.keras.layers.multiply()([gating, alarm_and_ones])
        # outputs = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(prod)
        # final_model = Model(inputs, outputs)

        vprint(self.verbose, f"\nARGUE networks built succesfully - properties: \n"
                             f"  > Input dimension: {self.input_dim}\n"
                             f"  > Encoder activations: {self.encoder_activation_dim}\n"
                             f"  > Decoder activations: {self.decoder_activation_dim}\n"
                             f"  > Latent dimension: {self.latent_dim}\n"
                             f"  > Number of decoders: {len(self.autoencoder_dict)}\n")
        return self

    def fit(self,
            x: Union[DataFrame, np.ndarray],
            partition_labels: Union[DataFrame, List[int]],
            validation_split: float = 0.1,
            batch_size: Optional[int] = 128,
            autoencoder_batch_size: Optional[int] = None,
            alarm_gating_batch_size: Optional[int] = None,
            epochs: Optional[int] = 100,
            autoencoder_epochs: Optional[int] = None,
            alarm_gating_epochs: Optional[int] = None,
            n_noise_samples: Optional[int] = None,
            noise_stdevs_away: float = 3.0,
            noise_stdev: float = 1.0,
            autoencoder_decay_after_epochs: Optional[int] = None,
            alarm_gating_decay_after_epochs: Optional[int] = None,
            decay_rate: Optional[float] = 0.7,  # 0.1 = heavy reduction, 0.9 = slight reduction
            optimizer: Union[tf.keras.optimizers.Optimizer, str] = "adam",
            fp_penalty: float = 0.1,
            fn_penalty: float = 0):

        autoencoder_epochs = epochs if autoencoder_epochs is None else autoencoder_epochs
        alarm_gating_epochs = epochs if alarm_gating_epochs is None else alarm_gating_epochs
        autoencoder_batch_size = batch_size if autoencoder_batch_size is None else autoencoder_batch_size
        alarm_gating_batch_size = batch_size if alarm_gating_batch_size is None else alarm_gating_batch_size

        # form initial training data making sure labels and partitions are right
        unique_partitions = np.unique(list(partition_labels))
        vprint(self.verbose, "Preparing data: slicing into partitions and batches...\n"
                             f"Data dimensions: {x.shape}")
        x_copy = x.copy()
        x_copy = pd.concat([x_copy, partition_labels], axis=1)
        x_copy.rename(columns={x_copy.columns[-1]: "partition"})

        # make gaussian noise samples so the optimization doesn't only see "healthy" data
        # and hence just learns to always predict healthy, i.e. P(healthy) = certain
        x_noise = generate_noise_samples(x_copy.drop(columns=["partition"]),
                                         quantiles=[0.005, 0.995], stdev=noise_stdev,
                                         stdevs_away=noise_stdevs_away, n_noise_samples=n_noise_samples)
        x_noise["partition"] = -1
        x_with_noise_and_labels = pd.concat([x_copy, x_noise]).reset_index(drop=True)
        x_with_noise_and_labels = shuffle(x_with_noise_and_labels)

        # get one hot encodings of the partitions to use as labels for the gating network
        gating_label_vectors = pd.get_dummies(x_with_noise_and_labels["partition"]).values

        # split into train and validation sets
        x_train, x_val, gating_train_labels, gating_val_labels = train_test_split(x_with_noise_and_labels,
                                                                                  gating_label_vectors,
                                                                                  test_size=validation_split)
        val_batch_size = 1024 if x_val.shape[0] >= 1024 else 128

        # make training set for the alarm and gating networks
        alarm_gating_train_dataset = tf.data.Dataset.from_tensor_slices(
            (x_train.drop(columns="partition"), gating_train_labels))
        alarm_gating_train_dataset = alarm_gating_train_dataset.shuffle(1024).batch(alarm_gating_batch_size,
                                                                                    drop_remainder=True).prefetch(2)
        alarm_gating_val_dataset = tf.data.Dataset.from_tensor_slices(
            (x_val.drop(columns="partition"),
             gating_val_labels))
        alarm_gating_val_dataset = alarm_gating_val_dataset.batch(val_batch_size, drop_remainder=True).prefetch(2)

        # make training set for the autoencoder pairs
        autoencoder_train_dataset_dict = {}
        autoencoder_val_dataset_dict = {}
        for partition_number, data_partition in enumerate(unique_partitions):
            train_dataset = x_train[x_train["partition"] == data_partition].drop(columns=["partition"])
            val_dataset = x_val[x_val["partition"] == data_partition].drop(columns=["partition"])
            train_dataset = shuffle(train_dataset)
            val_dataset = shuffle(val_dataset)
            val_batch_size = 1024 if val_dataset.shape[0] >= 1024 else 128

            ae_number_of_batches = train_dataset.shape[0] // autoencoder_batch_size
            alarm_gating_number_of_batches = x_train.shape[0] // alarm_gating_batch_size

            train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset)
            val_dataset = tf.data.Dataset.from_tensor_slices(val_dataset)

            train_dataset = train_dataset.shuffle(1024).batch(autoencoder_batch_size,
                                                              drop_remainder=True).prefetch(2)
            val_dataset = val_dataset.batch(val_batch_size, drop_remainder=True).prefetch(2)
            autoencoder_train_dataset_dict[f"partition_{data_partition}"] = train_dataset
            autoencoder_val_dataset_dict[f"partition_{data_partition}"] = val_dataset

            vprint(self.verbose, f"Autoencoder data partition {partition_number} batch size: "
                                 f"{autoencoder_batch_size}, number of batches (train set): {ae_number_of_batches}")

        vprint(self.verbose, f"Alarm-gating batch size: {alarm_gating_batch_size}, "
                             f"number of batches (train set): {alarm_gating_number_of_batches}")

        # init loss and metric functions
        ae_loss_fn, alarm_loss_fn, gating_loss_fn, final_loss_fn = self._init_loss_functions()
        ae_metric_fn, alarm_metric_fn, gating_metric_fn, final_metric_fn = self._init_metric_functions()

        # first train encoder and decoders
        vprint(self.verbose, "\n\n=== Phase 1: training autoencoder pairs ===")
        ae_optimizer = tf.keras.optimizers.get(optimizer)
        if autoencoder_decay_after_epochs is not None:
            decay_steps = alarm_gating_batch_size * autoencoder_decay_after_epochs * self.number_of_decoders
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.001,
                                                                         decay_steps=decay_steps,
                                                                         decay_rate=decay_rate)
            ae_optimizer.__init__(learning_rate=lr_schedule)

        @tf.function
        def _ae_train_step(x_batch_train):
            with tf.GradientTape() as tape:
                predictions = ae_model(x_batch_train, training=True)
                train_loss_value = ae_loss_fn(x_batch_train, predictions)
            gradients = tape.gradient(train_loss_value, ae_model.trainable_variables)
            ae_optimizer.apply_gradients(zip(gradients, ae_model.trainable_variables))
            ae_metric_fn.update_state(x_batch_train, predictions)
            return train_loss_value

        # train loop
        for epoch in range(1, autoencoder_epochs + 1):
            vprint(self.verbose, f"\n>> Epoch {epoch} - autoencoder")
            avg_ae_model_train_loss = []
            avg_ae_model_train_metric = []
            avg_ae_model_val_loss = []
            avg_ae_model_val_metric = []

            # weight update loop
            for ae_model_name, ae_model in self.autoencoder_dict.items():
                epoch_train_loss = []
                epoch_train_metric = []
                epoch_val_loss = []
                epoch_val_metric = []
                vprint(self.verbose > 1, f"== Model: {ae_model_name}, training steps:")

                # train loop:
                # for each model, iterate over all its batches from its own dataset to update weights
                # TODO in the future, if this doesnt work well, training should alternate between models,
                #  one batch for each at a time

                partition = ae_model_name[12:]
                for step, x_batch_train in enumerate(autoencoder_train_dataset_dict[f"partition_{partition}"]):
                    ae_train_loss = _ae_train_step(x_batch_train)
                    error_metric = ae_metric_fn.result()
                    epoch_train_metric.append(error_metric)
                    ae_metric_fn.reset_states()

                    ae_train_loss = float(ae_train_loss)
                    epoch_train_loss.append(ae_train_loss)
                    if step % 10 == 0 and self.verbose > 1:
                        print(f"Step {step} training loss: {ae_train_loss:.4f}, "
                              f"MAE: {float(error_metric):.4f}")

                # validation loop for the submodel
                for x_batch_val in autoencoder_val_dataset_dict[f"partition_{partition}"]:
                    ae_loss_value = self._ae_validation_step(x_batch_val, ae_model, ae_loss_fn, ae_metric_fn)
                    epoch_val_loss.append(ae_loss_value)

                    # NOTE: might need to be unindented
                    val_metric = ae_metric_fn.result()
                    epoch_val_metric.append(val_metric)
                    ae_metric_fn.reset_states()

                vprint(self.verbose > 1, f"Model {ae_model_name[-1]} loss [train: {np.mean(epoch_train_loss):.4f}, "
                                         f"val: {np.mean(epoch_val_loss):.4f}] "
                                         f"| MAE [train: {np.mean(epoch_train_metric):.4f}, "
                                         f"val: {np.mean(epoch_val_metric):.4f}]")
                avg_ae_model_train_loss.append(np.mean(epoch_train_loss))
                avg_ae_model_val_loss.append(np.mean(epoch_val_loss))
                avg_ae_model_train_metric.append(np.mean(epoch_train_metric))
                avg_ae_model_val_metric.append(np.mean(epoch_val_metric))
            vprint(self.verbose, f"--- Average epoch loss [train: {np.mean(avg_ae_model_train_loss):.4f}, "
                                 f"val: {np.mean(avg_ae_model_val_loss):.4f}] "
                                 f"| Average model MAE [train: {np.mean(avg_ae_model_train_metric):.4f}, "
                                 f"val: {np.mean(avg_ae_model_val_metric):.4f}]")
            if autoencoder_decay_after_epochs is not None:
                vprint(self.verbose and epoch % autoencoder_decay_after_epochs == 0,
                       "\nReduced learning rate!\n")

        self._make_non_trainable("autoencoders")

        # train alarm network
        vprint(self.verbose, "\n\n=== Phase 2: training alarm & gating networks ===")

        # init optimizers
        alarm_gating_optimizer = tf.keras.optimizers.get(optimizer)
        if alarm_gating_decay_after_epochs is not None:
            decay_steps = alarm_gating_batch_size * alarm_gating_decay_after_epochs
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.0001,
                                                                         decay_steps=decay_steps,
                                                                         decay_rate=decay_rate)
            alarm_gating_optimizer.__init__(learning_rate=lr_schedule)

        # training loop
        gating_model = self.input_to_gating
        alarm_model = self.input_to_alarm
        gating_network_variables = self.gating.keras_model.trainable_variables
        alarm_network_variables = self.alarm.keras_model.trainable_variables

        N_decoders = self.number_of_decoders

        @tf.function
        def _alarm_gating_train_step(x_batch_train, true_gating):
            # x_batch_train = x_batch_train[:1, ]
            # true_gating = true_gating[:1, ]

            true_alarm = 1 - true_gating[:, 1:]
            true_final = tf.cast(tf.reduce_min(true_alarm, axis=1), "float32")
            # update alarm model
            with tf.GradientTape() as tape:
                predicted_alarm = alarm_model(x_batch_train, training=True)
                if type(predicted_alarm) == list:
                    predicted_alarm = tf.keras.layers.concatenate(alarm_model(x_batch_train, training=True))
                    # predicted_alarm = tf.stack(alarm_model(x_batch_train, training=True), axis=2)
                    # predicted_alarm = tf.reshape(predicted_alarm, shape=(-1, N_decoders))
                alarm_loss_value = alarm_loss_fn(true_alarm, predicted_alarm)
            gradients = tape.gradient(alarm_loss_value, alarm_network_variables)
            alarm_gating_optimizer.apply_gradients(zip(gradients, alarm_network_variables))

            # update gating model
            with tf.GradientTape() as tape:
                predicted_gating = gating_model(x_batch_train, training=True)
                gating_loss_value = gating_loss_fn(true_gating, predicted_gating)
            gradients = tape.gradient(gating_loss_value, gating_network_variables)
            alarm_gating_optimizer.apply_gradients(zip(gradients, gating_network_variables))

            # update the whole model
            with tf.GradientTape() as tape:
                pred_alarm = alarm_model(x_batch_train, training=True)
                pred_alarm = tf.keras.layers.concatenate(pred_alarm)
                pred_alarm = tf.keras.layers.concatenate([tf.ones((256, 1)), pred_alarm])
                pred_gating = gating_model(x_batch_train, training=True)
                pred_prod = tf.keras.layers.multiply([pred_gating, pred_alarm])
                predicted_final = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(pred_prod)

                # predicted_alarm = tf.stack(alarm_model(x_batch_train, training=False), axis=2)
                # predicted_alarm = tf.reshape(predicted_alarm, shape=(-1, N_decoders))
                # alarm_and_ones = tf.concat([tf.ones((predicted_alarm.shape[0], 1)), predicted_alarm], axis=1)
                # predicted_gating = gating_model(x_batch_train, training=False)
                # predicted_prod = tf.multiply(predicted_gating, alarm_and_ones)
                # predicted_final = tf.reduce_sum(predicted_prod, axis=1)

                false_positive_loss = tf.reduce_mean(tf.keras.activations.relu(predicted_final - true_final))
                false_negative_loss = tf.reduce_mean(tf.keras.activations.relu(true_final - predicted_final))
                final_loss_value = final_loss_fn(true_final, predicted_final) + \
                                   fp_penalty * false_positive_loss + fn_penalty * false_negative_loss
            gradients = tape.gradient(final_loss_value, alarm_network_variables + gating_network_variables)
            alarm_gating_optimizer.apply_gradients(zip(gradients,
                                                       alarm_network_variables + gating_network_variables))

            # control false positve penalty using max(0, p-y) = relu(p-y) as regularizing term
            # total_loss = alarm_loss_value + gating_loss_value + final_loss_value + \
            #              fp_penalty * false_positive_loss + fn_penalty * false_negative_loss

            alarm_metric_fn.update_state(true_alarm, predicted_alarm)
            gating_metric_fn.update_state(true_gating, predicted_gating)
            final_metric_fn.update_state(true_final, predicted_final)
            total_loss = tf.constant(0)
            # final_loss_value = tf.constant(0)
            return total_loss, alarm_loss_value, gating_loss_value, final_loss_value

        # alarm & gating training loop
        for epoch in range(1, alarm_gating_epochs + 1):
            vprint(self.verbose, f"\n>> Epoch {epoch} - alarm & gating")
            alarm_epoch_train_loss = []
            alarm_epoch_train_metric = []
            alarm_epoch_val_loss = []
            alarm_epoch_val_metric = []
            gating_epoch_train_loss = []
            gating_epoch_train_metric = []
            gating_epoch_val_loss = []
            gating_epoch_val_metric = []
            final_epoch_train_loss = []
            final_epoch_train_metric = []
            final_epoch_val_loss = []
            final_epoch_val_metric = []

            # weight update loop
            for step, (x_batch_train, true_gating) in enumerate(alarm_gating_train_dataset):
                _, alarm_loss_value, gating_loss_value, final_loss_value = _alarm_gating_train_step(
                    x_batch_train, true_gating)
                alarm_epoch_train_loss.append(alarm_loss_value)
                gating_epoch_train_loss.append(gating_loss_value)
                final_epoch_train_loss.append(final_loss_value)

                alarm_epoch_train_metric.append(alarm_metric_fn.result())
                gating_epoch_train_metric.append(gating_metric_fn.result())
                final_epoch_train_metric.append(final_metric_fn.result())
                alarm_metric_fn.reset_states()
                gating_metric_fn.reset_states()
                final_metric_fn.reset_states()

                vprint(step % 20 == 0 and self.verbose > 1,
                       f"\nStep: {step} - "
                       f"Batch loss - Alarm: {float(alarm_loss_value):.4f}, "
                       f"Gating: {float(gating_loss_value):.4f}")

            # end of epoch validation loop
            for (x_batch_val, true_gating) in alarm_gating_val_dataset:
                _, alarm_loss_value, gating_loss_value, final_loss_value = \
                    self._alarm_gating_validation_step(x_batch_val, true_gating, alarm_loss_fn,
                                                       gating_loss_fn, final_loss_fn, alarm_metric_fn,
                                                       gating_metric_fn, final_metric_fn)
                alarm_epoch_val_loss.append(alarm_loss_value)
                gating_epoch_val_loss.append(gating_loss_value)
                final_epoch_val_loss.append(final_loss_value)

                alarm_epoch_val_metric.append(alarm_metric_fn.result())
                gating_epoch_val_metric.append(gating_metric_fn.result())
                final_epoch_val_metric.append(final_metric_fn.result())
                alarm_metric_fn.reset_states()
                gating_metric_fn.reset_states()
                final_metric_fn.reset_states()

            vprint(self.verbose, f"Alarm loss  [train: {np.mean(alarm_epoch_train_loss):.4f}, "
                                 f"val: {np.mean(alarm_epoch_val_loss):.4f}] "
                                 f"| MAE  [train: {np.mean(alarm_epoch_train_metric):.4f}, "
                                 f"val: {np.mean(alarm_epoch_val_metric):.4f}]")
            vprint(self.verbose, f"Gating loss [train: {np.mean(gating_epoch_train_loss):.4f}, "
                                 f"val: {np.mean(gating_epoch_val_loss):.4f}] "
                                 f"| MAE  [train: {np.mean(gating_epoch_train_metric):.4f}, "
                                 f"val: {np.mean(gating_epoch_val_metric):.4f}]")
            vprint(self.verbose, f"Final loss  [train: {np.mean(final_epoch_train_loss):.4f}, "
                                 f"val: {np.mean(final_epoch_val_loss):.4f}] "
                                 f"| MAE  [train: {np.mean(final_epoch_train_metric):.4f}, "
                                 f"val: {np.mean(final_epoch_val_metric):.4f}]"
                   )
            if alarm_gating_decay_after_epochs is not None:
                vprint(self.verbose and epoch % alarm_gating_decay_after_epochs == 0,
                       "\nReduced learning rate!\n")
        self._make_non_trainable("alarm")
        self._make_non_trainable("gating")

        vprint(self.verbose, "\n----------- Model fitted!\n\n")

    def predict(self, x: DataFrame):
        gating_vector = self.predict_gating_weights(x)
        alarm_vector = self.predict_alarm_probabilities(x)
        # predictions = alarm_vector.min(axis=0)

        # stack the virtual decision vector on top of the alarm verdicts vector
        alarm_vector = np.vstack([np.ones((1, x.shape[0])), alarm_vector]).transpose()

        # compute final weighted average anomaly score
        predictions = np.multiply(gating_vector, alarm_vector).sum(axis=1)
        return predictions

    def predict_gating_weights(self, x):
        return self.input_to_gating.predict(x).reshape((-1, self.number_of_decoders + 1))

    def predict_alarm_probabilities(self, x: DataFrame):
        alarm_vector = self.input_to_alarm.predict(x)
        alarm_vector = np.stack(alarm_vector).reshape((self.number_of_decoders, -1))
        return alarm_vector

    def predict_plot_anomalies(self,
                               x,
                               true_partitions: Optional[List[int]] = None,
                               window_length: Optional[Union[int, List[int]]] = None,
                               **kwargs):
        df_preds = pd.DataFrame(self.predict(x), columns=["Anomaly probability"])
        if x.index is not None:
            df_preds.index = x.index

        if window_length is not None:
            window_length = [window_length] if type(window_length) != list else window_length
            fig, ax = plt.subplots(1, 1)
            ax.plot(df_preds, label="Anomaly probability", alpha=0.6)
            ax.xaxis.set_major_locator(ticker.LinearLocator(numticks=8))
            plt.xticks(rotation=15)

            for window in window_length:
                df_MA = df_preds.rolling(window=window).mean()
                col_name = f"{window} sample moving average"
                df_MA.columns.values[0] = col_name
                ax.plot(df_MA, label=col_name)
            plt.legend()
            return fig, ax

        if true_partitions:
            df_preds["partition"] = true_partitions

        df_preds.index = pd.to_datetime(df_preds.index)
        df_preds.index = df_preds.index.map(lambda t: t.strftime("%d-%m-%Y"))
        fig = df_preds.plot(subplots=True, rot=15, color=["#4099DA", "red"], **kwargs)
        plt.xlabel("")
        plt.suptitle("ARGUE anomaly predictions")
        return fig

    def predict_reconstructions(self,
                                x: DataFrame):
        """
        Predicts reconstructions from the autoencoder pairs. The most suited autoencoder pair for a
        particular datapoint is inferred by the gating network, so the decoder with the highest computed
        gating weight is selected for each datapoint.

        :param x: Dataframe with feature observations
        :return: Dataframe with reconstructed features
        """

        # determine the decoder best suited for reconstructing each datapoint and only choose
        # that one's predictions/reconstructions
        best_decoder = self.predict_gating_weights(x)[:, 1:].argmax(axis=1)
        row_number = np.arange(best_decoder.shape[0])
        reconstructions = np.array([model.predict(x) for _, model in self.autoencoder_dict.items()])
        final_reconstructions = reconstructions[best_decoder, row_number, :]
        final_reconstructions = pd.DataFrame(final_reconstructions,
                                             columns=x.columns,
                                             index=x.index)
        return final_reconstructions

    def predict_plot_reconstructions(self,
                                     x: DataFrame,
                                     cols_to_plot: Optional[List[str]] = None,
                                     **kwargs):
        df_predictions = self.predict_reconstructions(x)
        col_names = x.columns
        col_names_pred = col_names + "_pred"
        df_predictions.columns = col_names_pred
        df_all = pd.concat([x, df_predictions], 1)

        swapped_col_order = []
        for i in range(len(col_names)):
            swapped_col_order.append(col_names[i])
            swapped_col_order.append(col_names_pred[i])

        df_all = df_all[swapped_col_order]
        if cols_to_plot is None:
            N_cols_to_plot = len(col_names) if len(col_names) <= 6 else 6
            cols_to_plot = df_all.columns.values[0: 2 * N_cols_to_plot]

        df_plots = df_all[cols_to_plot]

        graphs_in_same_plot = len(col_names) == len(col_names_pred)
        if graphs_in_same_plot:
            num_plots = int(df_plots.shape[1] / 2)
            fig, axes = plt.subplots(num_plots, 1, sharex=True)
            for axis, col in zip(np.arange(num_plots), np.arange(0, df_plots.shape[1], 2)):
                df_to_plot = df_plots.iloc[:, col: col + 2]
                df_to_plot.columns = ["Actual", "Predicted"]
                df_to_plot.index = pd.to_datetime(df_to_plot.index)
                df_to_plot.index = df_to_plot.index.map(lambda t: t.strftime("%d-%m-%Y"))
                df_to_plot.plot(ax=axes[axis], rot=15, legend=False)
                axes[axis].set_title(df_plots.columns[col], size=10)
                axes[axis].get_xaxis().get_label().set_visible(False)
            box = axes[axis].get_position()
            axes[axis].set_position(
                [box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9]
            )
            plt.legend(
                bbox_to_anchor=(0.7, -0.01),
                loc="lower right",
                bbox_transform=fig.transFigure,
                ncol=2,
                fancybox=True,
                shadow=True,
            )
            plt.suptitle("Model predictions")
            fig.tight_layout()
        else:
            for key, value in kwargs.items():
                df_all[key] = kwargs[key]
                cols_to_plot.append(key)
            fig = df_all[cols_to_plot].plot(subplots=True)
        return fig

    def save(self, path: Union[WindowsPath, str] = None, model_name: str = None):
        def _save_models_in_dict(model_dict: Dict):
            for name, model in model_dict.items():
                model.save(path / name)

        vprint(self.verbose, "\nSaving model...\n")
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
        print("\nLoading model...\n")
        model_name = model_name if model_name else "argue"
        path = get_model_archive_path() / model_name if not path else path

        # finally, load the dictionary storing the builtin/simple types, e.g. ints
        with open(path / "non_model_attributes.pkl", "rb") as file:
            non_model_attributes_dict = pickle.load(file)
        for name, attribute in non_model_attributes_dict.items():
            vars(self)[name] = attribute

        # an untrained model needs to be built before we can start loading it
        self.verbose = False
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

        print("... Model loaded and ready!")

        return self
