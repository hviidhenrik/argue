import datetime
import time
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import tensorflow as tf
import wandb
from pandas import DataFrame
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

from argue.models.base_model import BaseModel
from argue.utils.misc import (check_alarm_sparsity, generate_noise_samples,
                              plot_learning_schedule, vprint)
from argue.utils.model import Network

plt.style.use("seaborn")


class ARGUE(BaseModel):
    def __init__(
        self,
        input_dim: int = 3,
        number_of_decoders: int = 2,
        latent_dim: int = 2,
        verbose: int = 1,
        model_name: str = "",
    ):
        self.input_dim = input_dim
        self.number_of_decoders = number_of_decoders
        self.latent_dim = latent_dim
        self.encoder = None
        self.decoder_dict = {}
        self.alarm = None
        self.gating = None
        self.encoder_activation_dim = None
        self.decoder_activation_dim = None
        self.input_to_decoders = None
        self.input_to_alarm = None
        self.input_to_gating = None
        self.history = None
        self.verbose = verbose
        self.ae_train_loss = []
        self.ae_val_loss = []
        self.ae_train_metric = []
        self.ae_val_metric = []
        self.alarm_train_loss = []
        self.alarm_val_loss = []
        self.alarm_train_metric = []
        self.alarm_val_metric = []
        self.alarm_train_metric = []
        self.alarm_val_metric = []
        self.gating_train_loss = []
        self.gating_val_loss = []
        self.gating_train_metric = []
        self.gating_val_metric = []
        super().__init__(model_name=model_name)

    def _connect_autoencoder_pair(self, decoder):
        decoder_number = decoder.name[8:]
        inputs = self.encoder.keras_model.input
        x = self.encoder.keras_model(inputs)
        outputs = decoder.keras_model(x)
        return Model(inputs, outputs, name=f"autoencoder_{decoder_number}")

    def _connect_alarm_pair(self, decoder, use_encoder_activations: bool = False):
        decoder_number = decoder.name[8:]
        inputs = self.encoder.keras_model.input
        x = self.encoder.keras_model(inputs)
        x = decoder.activation_model(x)
        if use_encoder_activations:
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
    def _alarm_gating_validation_step(
        self, x_batch_val, true_gating, alarm_loss, gating_loss, final_loss, alarm_metric, gating_metric, final_metric,
    ):
        true_alarm = 1 - true_gating[:, 1:]
        true_final = tf.reduce_min(true_alarm, axis=1)
        predicted_alarm = tf.keras.layers.concatenate(self.input_to_alarm(x_batch_val, training=False))
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
        return (
            alarm_loss_value,
            gating_loss_value,
            final_loss_value,
            true_alarm,
            predicted_alarm,
        )

    @staticmethod
    def _init_loss_functions():
        ae_loss = tf.losses.BinaryCrossentropy()
        alarm_loss = tf.losses.BinaryCrossentropy()  # BinaryCrossentropy()
        gating_loss = tf.losses.CategoricalCrossentropy()
        final_loss = tf.losses.BinaryCrossentropy()
        return ae_loss, alarm_loss, gating_loss, final_loss

    @staticmethod
    def _init_metric_functions(final_output_metric: tf.metrics = tf.metrics.Accuracy()):
        ae_metric = tf.metrics.MeanAbsoluteError()
        alarm_metric = tf.metrics.MeanAbsoluteError()
        gating_metric = tf.metrics.MeanAbsoluteError()
        final_metric = final_output_metric  # tf.metrics.BinaryAccuracy()  # MeanAbsoluteError()
        return ae_metric, alarm_metric, gating_metric, final_metric

    @staticmethod
    def _init_optimizer(optimizer: str, learning_rate: float = 0.0003):
        optimizer = tf.keras.optimizers.get(optimizer)
        optimizer.__init__(learning_rate=learning_rate)
        return optimizer

    @staticmethod
    def _init_optimizer_with_lr_schedule(
        optimizer: str,
        initial_learning_rate: float = 0.0003,
        decay_after_epochs: int = 10,
        decay_rate: float = 0.7,
        dataset_rows: int = None,
        batch_size: int = None,
        total_epochs: int = 4,
        plot_schedule: bool = False,
    ):
        steps_per_epoch = dataset_rows // batch_size
        decay_steps = np.multiply(steps_per_epoch, decay_after_epochs)
        staircase = True
        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate, decay_steps, decay_rate, staircase=staircase
        )
        optimizer = tf.keras.optimizers.get(optimizer)
        optimizer.__init__(learning_rate=learning_rate)

        if plot_schedule:
            total_steps = total_epochs * steps_per_epoch
            plot_learning_schedule(
                total_steps=total_steps,
                initial_learning_rate=initial_learning_rate,
                decay_rate=decay_rate,
                decay_steps=decay_steps,
                staircase=staircase,
            )
            plt.show()
        return optimizer

    def _unfreeze_partition_decoder(self, partition):
        self.input_to_decoders.trainable = False
        self.encoder.keras_model.trainable = True
        decoder_to_train = int(partition[10:])
        decoder_model = self.input_to_decoders.layers[decoder_to_train + 1]
        decoder_name = decoder_model.name
        if "decoder" not in decoder_name:
            raise Exception(
                f'Wrong model name: "{decoder_name}" detected in decoder training! ' f'Should contain "decoder_<k>"'
            )
        decoder_model.trainable = True
        return decoder_model.trainable_variables

    def build_model(
        self,
        encoder_hidden_layers: List[int] = [10, 8, 5],
        decoders_hidden_layers: List[int] = [5, 8, 10],
        alarm_hidden_layers: List[int] = [15, 12, 10],
        gating_hidden_layers: List[int] = [15, 12, 10],
        encoder_activation: str = "tanh",
        decoders_activation: str = "tanh",
        alarm_activation: str = "tanh",
        gating_activation: str = "tanh",
        all_activations: Optional[str] = None,
        use_encoder_activations_in_alarm: bool = True,
        use_latent_activations_in_encoder_activations: bool = True,
        use_decoder_outputs_in_decoder_activations: bool = True,
        encoder_dropout_frac: Optional[float] = None,
        decoders_dropout_frac: Optional[float] = None,
        alarm_dropout_frac: Optional[float] = None,
        gating_dropout_frac: Optional[float] = None,
        make_model_visualiations: bool = False,
    ):
        self.hyperparameters = {
            "model_name": "ARGUE",
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "encoder_layers": encoder_hidden_layers,
            "decoder_layers": decoders_hidden_layers,
            "alarm_layers": alarm_hidden_layers,
            "gating_layers": gating_hidden_layers,
            "N_decoders": self.number_of_decoders,
            "N_encoder_activations": self.encoder_activation_dim,
            "N_decoder_activations": self.decoder_activation_dim,
            "encoder_activation": encoder_activation,
            "decoder_activation": decoders_activation,
            "alarm_activation": alarm_activation,
            "gating_activation": gating_activation,
            "use_encoder_activations_in_alarm": use_encoder_activations_in_alarm,
            "use_latent_activations_in_encoder_activations": use_latent_activations_in_encoder_activations,
            "use_decoder_outputs_in_decoder_activations": use_decoder_outputs_in_decoder_activations,
            "encoder_dropout_frac": encoder_dropout_frac,
            "decoder_dropout_frac": decoders_dropout_frac,
            "alarm_dropout_frac": alarm_dropout_frac,
            "gating_dropout_frac": gating_dropout_frac,
        }
        # if all_activations is specified, the same activation function is used in all hidden layers
        if all_activations is not None:
            encoder_activation = all_activations
            decoders_activation = all_activations
            alarm_activation: all_activations
            gating_activation: all_activations

        # build shared encoder
        self.encoder = Network(name="encoder").build_model(
            input_layer=Input(shape=(self.input_dim,)),
            output_layer=Dense(self.latent_dim, encoder_activation),
            units_in_layers=encoder_hidden_layers,
            activation=encoder_activation,
            dropout_frac=encoder_dropout_frac,
            keep_output_layer_activations=use_latent_activations_in_encoder_activations,
        )

        # set constants
        self.encoder_activation_dim = self.encoder.get_activation_dim()
        self.decoder_activation_dim = int(np.sum(decoders_hidden_layers))
        self.hyperparameters.update(
            {"N_encoder_activations": self.encoder_activation_dim, "N_decoder_activations": self.decoder_activation_dim}
        )
        if use_decoder_outputs_in_decoder_activations:
            self.decoder_activation_dim += self.input_dim

        # build alarm network
        alarm_input_dim = self.decoder_activation_dim
        if use_encoder_activations_in_alarm:
            alarm_input_dim += self.encoder_activation_dim
        self.alarm = Network(name="alarm").build_model(
            input_layer=Input(shape=(alarm_input_dim,)),
            output_layer=Dense(1, "sigmoid"),
            units_in_layers=alarm_hidden_layers,
            activation=alarm_activation,
            dropout_frac=alarm_dropout_frac,
        )

        self.gating = Network(name="gating").build_model(
            input_layer=Input(shape=(self.encoder_activation_dim,)),
            output_layer=Dense(self.number_of_decoders + 1, "softmax"),
            units_in_layers=gating_hidden_layers,
            activation=gating_activation,
            dropout_frac=gating_dropout_frac,
        )

        alarm_outputs = []
        decoder_outputs = []
        for i in range(1, self.number_of_decoders + 1):
            decoder_name = f"decoder_{i}"
            decoder = Network(name=decoder_name).build_model(
                input_layer=Input(shape=(self.latent_dim,)),
                output_layer=Dense(self.input_dim, "sigmoid"),
                units_in_layers=decoders_hidden_layers,
                activation=decoders_activation,
                dropout_frac=decoders_dropout_frac,
                keep_output_layer_activations=use_decoder_outputs_in_decoder_activations,
            )
            self.decoder_dict[decoder_name] = decoder
            # connect common encoder with each decoder
            decoder_output_tensor = self._connect_autoencoder_pair(decoder).output
            decoder_outputs.append(decoder_output_tensor)
            # connect encoder with alarm model through each decoder/expert network

            alarm_output_tensor = self._connect_alarm_pair(decoder, use_encoder_activations_in_alarm).output
            alarm_outputs.append(alarm_output_tensor)

        self.input_to_decoders = Model(inputs=self.encoder.keras_model.input, outputs=decoder_outputs)

        self.input_to_decoders.trainable = False
        self.encoder.keras_model.trainable = True

        self.input_to_alarm = Model(inputs=self.encoder.keras_model.input, outputs=alarm_outputs)
        self.input_to_gating = self._connect_gating()

        if make_model_visualiations:
            # if plot_model doesn't work, first pip install pydot, then pip install pydotplus, then go to:
            # https://graphviz.gitlab.io/download/ and download and install Graphviz. It must be added to
            # PATH environment variable in order to work since keras tries to call dot.exe. So
            # Graphviz\bin\ must be on the PATH.
            tf.keras.utils.plot_model(
                self.encoder.keras_model, to_file="encoder.png", show_shapes=True, show_layer_names=True,
            )
            tf.keras.utils.plot_model(
                self.encoder.activation_model,
                to_file="encoder_activations.png",
                show_shapes=True,
                show_layer_names=True,
            )
            tf.keras.utils.plot_model(
                self.decoder_dict["decoder_1"].keras_model,
                to_file="decoder.png",
                show_shapes=True,
                show_layer_names=True,
            )
            tf.keras.utils.plot_model(
                self.decoder_dict["decoder_1"].activation_model,
                to_file="decoder_activations.png",
                show_shapes=True,
                show_layer_names=True,
            )
            tf.keras.utils.plot_model(
                self.alarm.keras_model, to_file="alarm.png", show_shapes=True, show_layer_names=True,
            )
            tf.keras.utils.plot_model(
                self.gating.keras_model, to_file="gating.png", show_shapes=True, show_layer_names=True,
            )
            tf.keras.utils.plot_model(
                self.input_to_decoders, to_file="input_to_decoders.png", show_shapes=True, show_layer_names=True,
            )
            tf.keras.utils.plot_model(
                self.input_to_alarm, to_file="input_to_alarm.png", show_shapes=True, show_layer_names=True,
            )
            tf.keras.utils.plot_model(
                self.input_to_gating, to_file="input_to_gating.png", show_shapes=True, show_layer_names=True,
            )

        vprint(
            self.verbose,
            f"\nARGUE networks built succesfully - properties: \n"
            f"  > Input dimension: {self.input_dim}\n"
            f"  > Encoder activations: {self.encoder_activation_dim}\n"
            f"  > Decoder activations: {self.decoder_activation_dim}\n"
            f"  > Latent dimension: {self.latent_dim}\n"
            f"  > Number of decoders: {self.number_of_decoders}\n",
        )
        return self

    def fit(
        self,
        x: Union[DataFrame, np.ndarray],
        partition_labels: pd.Series,
        validation_split: float = 0.1,
        batch_size: Optional[int] = 128,
        autoencoder_batch_size: Optional[int] = None,
        alarm_gating_batch_size: Optional[int] = None,
        epochs: Optional[int] = 100,
        autoencoder_epochs: Optional[int] = None,
        alarm_gating_epochs: Optional[int] = None,
        n_noise_samples: Optional[int] = None,
        noise_stdevs_away: float = 3.0,
        noise_mean: float = 0.5,
        noise_stdev: float = 1.0,
        ae_learning_rate: Union[float, List[float]] = 0.0001,
        alarm_gating_learning_rate: float = 0.0001,
        autoencoder_decay_after_epochs: Optional[Union[int, List[int]]] = None,
        alarm_decay_after_epochs: Optional[int] = None,
        gating_decay_after_epochs: Optional[int] = None,
        decay_rate: Optional[float] = 0.7,  # 0.1 = heavy reduction, 0.9 = slight reduction
        optimizer: Union[tf.keras.optimizers.Optimizer, str] = "adam",
        fp_penalty: float = 0,
        fp_tolerance: float = 0.3,
        fn_penalty: float = 0,
        fn_tolerance: float = 0.3,
        plot_normal_vs_noise: bool = False,
        plot_learning_rate_decay: bool = False,
        log_with_wandb: bool = False,
        final_output_metric: tf.metrics = tf.metrics.Accuracy(),
    ):
        self.hyperparameters.update(
            {
                "validation_split": validation_split,
                "autoencoder_epochs": autoencoder_epochs,
                "alarm_epochs": alarm_gating_epochs,
                "autoencoder_learning_rate": ae_learning_rate,
                "alarm_learning_rate": alarm_gating_learning_rate,
                "autoencoder_batch_size": autoencoder_batch_size,
                "alarm_batch_size": alarm_gating_batch_size,
                "n_noise_samples": n_noise_samples,
                "noise_stdevs_away": noise_stdevs_away,
                "noise_mean": noise_mean,
                "noise_stdev": noise_stdev,
                "autoencoder_decay_after_epochs": autoencoder_decay_after_epochs,
                "alarm_decay_after_epochs": alarm_decay_after_epochs,
                "gating_decay_after_epochs": gating_decay_after_epochs,
                "decay_rate": decay_rate,
                "fp_penalty": fp_penalty,
                "fp_tolerance": fp_tolerance,
                "fn_penalty": fp_penalty,
                "fn_tolerance": fp_tolerance,
            }
        )

        if log_with_wandb:
            wandb.init(config=self.hyperparameters)

        autoencoder_epochs = epochs if autoencoder_epochs is None else autoencoder_epochs
        alarm_gating_epochs = epochs if alarm_gating_epochs is None else alarm_gating_epochs
        autoencoder_batch_size = batch_size if autoencoder_batch_size is None else autoencoder_batch_size
        alarm_gating_batch_size = batch_size if alarm_gating_batch_size is None else alarm_gating_batch_size

        # form initial training data making sure labels and partitions are right
        unique_partitions = np.unique(list(partition_labels))
        vprint(
            self.verbose, "Preparing data: slicing into partitions and batches...\n" f"Data dimensions: {x.shape}",
        )
        x_copy = x.copy()
        x_copy = pd.concat([x_copy, partition_labels], axis=1)
        x_copy = x_copy.rename(columns={x_copy.columns[-1]: "partition"})

        # make gaussian noise samples so the optimization doesn't only see "healthy" data
        # and hence just learns to always predict healthy, i.e. P(healthy) = certain
        x_noise = generate_noise_samples(
            x_copy.drop(columns=["partition"]), mean=noise_mean, stdev=noise_stdev, n_noise_samples=n_noise_samples,
        )

        if plot_normal_vs_noise:
            pca = PCA(2).fit(x_copy.drop(columns=["partition"]))
            pca_train = pca.transform(x_copy.drop(columns=["partition"]))
            pca_noise = pca.transform(x_noise)
            plt.scatter(pca_noise[:, 0], pca_noise[:, 1], s=5, label="noise data")
            plt.scatter(pca_train[:, 0], pca_train[:, 1], s=5, label="normal data")
            plt.suptitle("PCA of normal data vs generated noise")
            plt.legend()
            plt.show()

        x_noise["partition"] = -1
        x_with_noise_and_labels = pd.concat([x_copy, x_noise]).reset_index(drop=True)
        x_with_noise_and_labels = shuffle(x_with_noise_and_labels)

        # get one hot encodings of the partitions to use as labels for the gating network
        gating_label_vectors = pd.get_dummies(x_with_noise_and_labels["partition"]).values
        x_train, x_val, gating_train_labels, gating_val_labels = train_test_split(
            x_with_noise_and_labels, gating_label_vectors, test_size=validation_split
        )
        x_train = x_train.reset_index(drop=True)
        x_val = x_val.reset_index(drop=True)

        alarm_gating_val_batch_size = np.min([x_val.shape[0], 1024])
        alarm_gating_train_dataset = tf.data.Dataset.from_tensor_slices(
            (x_train.drop(columns="partition"), gating_train_labels)
        )
        alarm_gating_train_dataset = (
            alarm_gating_train_dataset.shuffle(1024).batch(alarm_gating_batch_size, drop_remainder=True).prefetch(2)
        )
        alarm_gating_val_dataset = tf.data.Dataset.from_tensor_slices(
            (x_val.drop(columns="partition"), gating_val_labels)
        )
        alarm_gating_val_dataset = alarm_gating_val_dataset.batch(
            alarm_gating_val_batch_size, drop_remainder=True
        ).prefetch(2)
        alarm_gating_number_of_batches = x_train.shape[0] // alarm_gating_batch_size

        autoencoder_train_dataset_dict = {}
        autoencoder_val_dataset_dict = {}
        autoencoder_data_partition_sizes = []
        for partition_number, data_partition in enumerate(unique_partitions):
            train_dataset = x_train[x_train["partition"] == data_partition].drop(columns=["partition"])
            val_dataset = x_val[x_val["partition"] == data_partition].drop(columns=["partition"])
            train_dataset = shuffle(train_dataset)
            val_dataset = shuffle(val_dataset)

            autoencoder_batch_size = np.min([train_dataset.shape[0], autoencoder_batch_size])
            ae_val_batch_size = np.min([val_dataset.shape[0], 1024])

            ae_number_of_batches = train_dataset.shape[0] // autoencoder_batch_size
            autoencoder_data_partition_sizes.append(train_dataset.shape[0])

            train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset)
            val_dataset = tf.data.Dataset.from_tensor_slices(val_dataset)

            train_dataset = train_dataset.shuffle(1024).batch(autoencoder_batch_size, drop_remainder=True).prefetch(2)
            val_dataset = val_dataset.batch(ae_val_batch_size, drop_remainder=True).prefetch(2)
            autoencoder_train_dataset_dict[f"partition_{data_partition}"] = train_dataset
            autoencoder_val_dataset_dict[f"partition_{data_partition}"] = val_dataset

            vprint(
                self.verbose,
                f"Autoencoder data partition {partition_number} batch size: "
                f"{autoencoder_batch_size}, number of batches (train set): {ae_number_of_batches}",
            )

        vprint(
            self.verbose,
            f"Alarm-gating batch size: {alarm_gating_batch_size}, "
            f"number of batches (train set): {alarm_gating_number_of_batches}",
        )

        # init loss and metric functions
        (ae_loss_fn, alarm_loss_fn, gating_loss_fn, final_loss_fn,) = self._init_loss_functions()
        (ae_metric_fn, alarm_metric_fn, gating_metric_fn, final_metric_fn,) = self._init_metric_functions(
            final_output_metric=final_output_metric
        )
        final_output_metric_name = str(final_output_metric.name)

        # first train encoder and decoders
        vprint(self.verbose, "\n\n=== Phase 1: training autoencoder pairs ===")
        if autoencoder_decay_after_epochs is None:
            ae_optimizer = self._init_optimizer(optimizer=optimizer, learning_rate=ae_learning_rate)
        else:
            ae_dataset_rows = self.number_of_decoders * np.max(autoencoder_data_partition_sizes)
            ae_optimizer = self._init_optimizer_with_lr_schedule(
                optimizer=optimizer,
                initial_learning_rate=ae_learning_rate,
                decay_after_epochs=autoencoder_decay_after_epochs,
                decay_rate=decay_rate,
                dataset_rows=ae_dataset_rows,
                batch_size=autoencoder_batch_size,
                total_epochs=autoencoder_epochs,
                plot_schedule=plot_learning_rate_decay,
            )

        # TODO tf.function throws error when run in "fwp30_argue.py", but not in the do_test_example.py
        # @tf.function
        def _ae_train_step(x_batch_train):
            with tf.GradientTape() as tape:
                predictions = ae_model(x_batch_train, training=True)
                train_loss_value = ae_loss_fn(x_batch_train, predictions)
            gradients = tape.gradient(train_loss_value, encoder_network_variables + decoder_variables)
            ae_optimizer.apply_gradients(zip(gradients, encoder_network_variables + decoder_variables))
            ae_metric_fn.update_state(x_batch_train, predictions)
            return train_loss_value

        # train loop
        ae_model = self.input_to_decoders
        encoder_network_variables = self.encoder.keras_model.trainable_variables
        for epoch in range(1, autoencoder_epochs + 1):
            epoch_start = time.time()
            vprint(self.verbose, f"\n>> Epoch {epoch} - autoencoder ")
            avg_ae_model_train_loss = []
            avg_ae_model_train_metric = []
            avg_ae_model_val_loss = []
            avg_ae_model_val_metric = []

            # weight update loop
            # TODO make naming of partitions standardized across datasets, or at least check if it matters
            for partition, ae_train_dataset in autoencoder_train_dataset_dict.items():
                ae_val_dataset = autoencoder_val_dataset_dict[partition]

                epoch_train_loss = []
                epoch_train_metric = []
                epoch_val_loss = []
                epoch_val_metric = []
                vprint(self.verbose > 1, f"== Model: {partition[10:]}, training steps:")

                # train loop:
                decoder_variables = self._unfreeze_partition_decoder(partition)
                for step, x_batch_train in enumerate(ae_train_dataset):
                    ae_train_loss = _ae_train_step(x_batch_train)
                    error_metric = ae_metric_fn.result()
                    epoch_train_metric.append(error_metric)
                    ae_metric_fn.reset_states()

                    ae_train_loss = float(ae_train_loss)
                    epoch_train_loss.append(ae_train_loss)
                    if step % 2 == 0 and self.verbose > 2:
                        print(f"Step {step} training loss: {ae_train_loss:.4f}, " f"MAE: {float(error_metric):.4f}")

                # validation loop for the submodel
                for x_batch_val in ae_val_dataset:
                    ae_loss_value = self._ae_validation_step(x_batch_val, ae_model, ae_loss_fn, ae_metric_fn)
                    epoch_val_loss.append(ae_loss_value)

                    # NOTE: might need to be unindented
                    val_metric = ae_metric_fn.result()
                    epoch_val_metric.append(val_metric)
                    ae_metric_fn.reset_states()

                    vprint(
                        self.verbose > 1,
                        f"Model {partition[10:]} loss [train: {np.mean(epoch_train_loss):.4f}, "
                        f"val: {np.mean(epoch_val_loss):.4f}] "
                        f"| MAE [train: {np.mean(epoch_train_metric):.4f}, "
                        f"val: {np.mean(epoch_val_metric):.4f}]",
                    )
                    avg_ae_model_train_loss.append(np.mean(epoch_train_loss))
                    avg_ae_model_val_loss.append(np.mean(epoch_val_loss))
                    avg_ae_model_train_metric.append(np.mean(epoch_train_metric))
                    avg_ae_model_val_metric.append(np.mean(epoch_val_metric))

            vprint(
                self.verbose,
                f"--- Average epoch loss [train: {np.mean(avg_ae_model_train_loss):.4f}, "
                f"val: {np.mean(avg_ae_model_val_loss):.4f}] "
                f"| Average model MAE [train: {np.mean(avg_ae_model_train_metric):.4f}, "
                f"val: {np.mean(avg_ae_model_val_metric):.4f}]",
            )

            self.ae_train_loss.append(np.mean(avg_ae_model_train_loss))
            self.ae_val_loss.append(np.mean(avg_ae_model_val_loss))
            self.ae_train_metric.append(np.mean(avg_ae_model_train_metric))
            self.ae_val_metric.append(np.mean(avg_ae_model_val_metric))

            if log_with_wandb:
                wandb.log(
                    {
                        "ae_avg_train_loss": np.mean(avg_ae_model_train_loss),
                        "ae_avg_val_loss": np.mean(avg_ae_model_val_loss),
                        "ae_avg_train_MAE": np.mean(avg_ae_model_train_metric),
                        "ae_avg_val_MAE": np.mean(avg_ae_model_val_metric),
                    }
                )

            epoch_end = time.time()
            epoch_time_elapsed = epoch_end - epoch_start
            vprint(self.verbose, f"--- Time elapsed: {epoch_time_elapsed:.2f} seconds")
            if autoencoder_decay_after_epochs is not None:
                vprint(
                    self.verbose and epoch % autoencoder_decay_after_epochs == 0, "\nReduced learning rate!\n",
                )

        self._make_non_trainable("autoencoders")

        # train alarm network
        vprint(self.verbose, "\n\n=== Phase 2: training alarm & gating networks ===")

        # init optimizers
        if alarm_decay_after_epochs is None:
            alarm_optimizer = self._init_optimizer(optimizer=optimizer, learning_rate=alarm_gating_learning_rate)
            final_optimizer = self._init_optimizer(optimizer=optimizer, learning_rate=alarm_gating_learning_rate)
        else:
            alarm_optimizer = self._init_optimizer_with_lr_schedule(
                optimizer=optimizer,
                initial_learning_rate=alarm_gating_learning_rate,
                decay_after_epochs=alarm_decay_after_epochs,
                decay_rate=decay_rate,
                dataset_rows=x_train.shape[0],
                batch_size=alarm_gating_batch_size,
                total_epochs=alarm_gating_epochs,
                plot_schedule=plot_learning_rate_decay,
            )
            final_optimizer = self._init_optimizer_with_lr_schedule(
                optimizer=optimizer,
                initial_learning_rate=alarm_gating_learning_rate,
                decay_after_epochs=alarm_decay_after_epochs,
                decay_rate=decay_rate,
                dataset_rows=x_train.shape[0],
                batch_size=alarm_gating_batch_size,
                total_epochs=alarm_gating_epochs,
                plot_schedule=False,
            )
        if gating_decay_after_epochs is None:
            gating_optimizer = self._init_optimizer(optimizer=optimizer, learning_rate=alarm_gating_learning_rate)
        else:
            gating_optimizer = self._init_optimizer_with_lr_schedule(
                optimizer=optimizer,
                initial_learning_rate=alarm_gating_learning_rate,
                decay_after_epochs=gating_decay_after_epochs,
                decay_rate=decay_rate,
                dataset_rows=x_train.shape[0],
                batch_size=alarm_gating_batch_size,
                total_epochs=alarm_gating_epochs,
                plot_schedule=plot_learning_rate_decay,
            )

        # training loop
        gating_model = self.input_to_gating
        alarm_model = self.input_to_alarm
        gating_network_variables = self.gating.keras_model.trainable_variables
        alarm_network_variables = self.alarm.keras_model.trainable_variables

        @tf.function
        def _alarm_gating_train_step(x_batch_train, true_gating):
            true_alarm = 1 - true_gating[:, 1:]
            true_final = tf.cast(tf.reduce_min(true_alarm, axis=1), "float32")

            # update alarm model
            with tf.GradientTape() as tape:
                predicted_alarm = tf.keras.layers.concatenate(alarm_model(x_batch_train, training=True))
                alarm_loss_value = alarm_loss_fn(true_alarm, predicted_alarm)
            gradients = tape.gradient(alarm_loss_value, alarm_network_variables)
            alarm_optimizer.apply_gradients(zip(gradients, alarm_network_variables))

            # update gating model
            with tf.GradientTape() as tape:
                predicted_gating = gating_model(x_batch_train, training=True)
                gating_loss_value = gating_loss_fn(true_gating, predicted_gating)
            gradients = tape.gradient(gating_loss_value, gating_network_variables)
            gating_optimizer.apply_gradients(zip(gradients, gating_network_variables))

            # update the whole model
            with tf.GradientTape() as tape:
                predicted_alarm = alarm_model(x_batch_train, training=True)
                predicted_alarm = tf.keras.layers.concatenate(predicted_alarm)
                predicted_alarm_and_ones = tf.keras.layers.concatenate(
                    [tf.ones((predicted_alarm.shape[0], 1)), predicted_alarm]
                )
                predicted_gating = gating_model(x_batch_train, training=True)
                predicted_product = tf.keras.layers.multiply([predicted_gating, predicted_alarm_and_ones])
                predicted_final = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(predicted_product)

                # control false positve penalty using max(0, p-y) = relu(p-y) as regularizing term.
                # will only be positive for true_final = 0, and 0 if it's 1
                fp_term = tf.keras.activations.relu(predicted_final - true_final - fp_tolerance)
                false_positive_loss = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x))(fp_term)
                # will only be positive for true_final = 1, and 0 if it's 1
                # the tolerance is the acceptable prediction error before the penalty is applied,
                # e.g. if true_final = 1, predicted_final = 0.7 and fn_tolerance is 0.4, no penalty is applied
                fn_term = tf.keras.activations.relu(true_final - predicted_final - fn_tolerance)
                false_negative_loss = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x))(fn_term)
                final_loss_value = (
                    final_loss_fn(true_final, predicted_final)
                    + fp_penalty * false_positive_loss
                    + fn_penalty * false_negative_loss
                )
            gradients = tape.gradient(final_loss_value, alarm_network_variables + gating_network_variables)
            final_optimizer.apply_gradients(zip(gradients, alarm_network_variables + gating_network_variables))

            alarm_metric_fn.update_state(true_alarm, predicted_alarm)
            gating_metric_fn.update_state(true_gating, predicted_gating)
            final_metric_fn.update_state(true_final, predicted_final)
            return alarm_loss_value, gating_loss_value, final_loss_value

        # alarm & gating training loop
        for epoch in range(1, alarm_gating_epochs + 1):
            epoch_start = time.time()
            vprint(self.verbose, f"\n>> Epoch {epoch} - alarm & gating")
            alarm_epoch_train_loss = []
            alarm_epoch_train_metric = []
            alarm_epoch_val_loss = []
            alarm_epoch_val_metric = []
            alarm_epoch_sparsity = []
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
                (alarm_loss_value, gating_loss_value, final_loss_value,) = _alarm_gating_train_step(
                    x_batch_train, true_gating
                )
                alarm_epoch_train_loss.append(alarm_loss_value)
                gating_epoch_train_loss.append(gating_loss_value)
                final_epoch_train_loss.append(final_loss_value)

                alarm_epoch_train_metric.append(alarm_metric_fn.result())
                gating_epoch_train_metric.append(gating_metric_fn.result())
                final_epoch_train_metric.append(final_metric_fn.result())

                alarm_metric_fn.reset_states()
                gating_metric_fn.reset_states()
                final_metric_fn.reset_states()

                vprint(
                    step % 100 == 0 and self.verbose == 2,
                    f"\nStep: {step} - "
                    f"Batch loss - Alarm: {float(alarm_loss_value):.4f}, "
                    f"Gating: {float(gating_loss_value):.4f}",
                )
                vprint(
                    step % 20 == 0 and self.verbose > 2,
                    f"\nStep: {step} - "
                    f"Batch loss - Alarm: {float(alarm_loss_value):.4f}, "
                    f"Gating: {float(gating_loss_value):.4f}",
                )

            # end of epoch validation loop
            for (x_batch_val, true_gating) in alarm_gating_val_dataset:
                (
                    alarm_loss_value,
                    gating_loss_value,
                    final_loss_value,
                    true_alarm,
                    predicted_alarm,
                ) = self._alarm_gating_validation_step(
                    x_batch_val,
                    true_gating,
                    alarm_loss_fn,
                    gating_loss_fn,
                    final_loss_fn,
                    alarm_metric_fn,
                    gating_metric_fn,
                    final_metric_fn,
                )
                alarm_epoch_val_loss.append(alarm_loss_value)
                gating_epoch_val_loss.append(gating_loss_value)
                final_epoch_val_loss.append(final_loss_value)

                alarm_epoch_val_metric.append(alarm_metric_fn.result())
                gating_epoch_val_metric.append(gating_metric_fn.result())
                final_epoch_val_metric.append(final_metric_fn.result())
                alarm_epoch_sparsity.append(check_alarm_sparsity(true_alarm, predicted_alarm))
                alarm_metric_fn.reset_states()
                gating_metric_fn.reset_states()
                final_metric_fn.reset_states()

            self.alarm_train_loss.append(np.mean(alarm_epoch_train_loss))
            self.alarm_val_loss.append(np.mean(alarm_epoch_val_loss))
            self.alarm_train_metric.append(np.mean(alarm_epoch_train_metric))
            self.alarm_val_metric.append(np.mean(alarm_epoch_val_metric))

            self.gating_train_loss.append(np.mean(gating_epoch_train_loss))
            self.gating_val_loss.append(np.mean(gating_epoch_val_loss))
            self.gating_train_metric.append(np.mean(gating_epoch_train_metric))
            self.gating_val_metric.append(np.mean(gating_epoch_val_metric))

            if log_with_wandb:
                wandb.log(
                    {
                        "alarm_train_loss": np.mean(alarm_epoch_train_loss),
                        "alarm_val_loss": np.mean(alarm_epoch_val_loss),
                        "alarm_train_MAE": np.mean(alarm_epoch_train_metric),
                        "alarm_val_MAE": np.mean(alarm_epoch_val_metric),
                        "gating_train_loss": np.mean(gating_epoch_train_loss),
                        "gating_val_loss": np.mean(gating_epoch_val_loss),
                        "gating_train_MAE": np.mean(gating_epoch_train_metric),
                        "gating_val_MAE": np.mean(gating_epoch_val_metric),
                        "final_output_train_loss": np.mean(final_epoch_train_loss),
                        "final_output_val_loss": np.mean(final_epoch_val_loss),
                        f"final_output_train_{final_output_metric_name}": np.mean(final_epoch_train_metric),
                        f"final_output_val_{final_output_metric_name}": np.mean(final_epoch_val_metric),
                    }
                )

            vprint(
                self.verbose,
                f"--- Alarm loss  [train: {np.mean(alarm_epoch_train_loss):.4f}, "
                f"val: {np.mean(alarm_epoch_val_loss):.4f}] "
                f"| MAE      [train: {np.mean(alarm_epoch_train_metric):.4f}, "
                f"val: {np.mean(alarm_epoch_val_metric):.4f}, "
                f"sparsity: {np.mean(alarm_epoch_sparsity):.4f}]",
            )
            vprint(
                self.verbose,
                f"--- Gating loss [train: {np.mean(gating_epoch_train_loss):.4f}, "
                f"val: {np.mean(gating_epoch_val_loss):.4f}] "
                f"| MAE      [train: {np.mean(gating_epoch_train_metric):.4f}, "
                f"val: {np.mean(gating_epoch_val_metric):.4f}]",
            )
            vprint(
                self.verbose,
                f"--- Final loss  [train: {np.mean(final_epoch_train_loss):.4f}, "
                f"val: {np.mean(final_epoch_val_loss):.4f}] "
                f"| {final_output_metric_name} [train: {np.mean(final_epoch_train_metric):.4f}, "
                f"val: {np.mean(final_epoch_val_metric):.4f}]",
            )
            epoch_end = time.time()
            epoch_time_elapsed = epoch_end - epoch_start
            vprint(self.verbose, f"--- Time elapsed: {epoch_time_elapsed:.2f} seconds")

        self._make_non_trainable("alarm")
        self._make_non_trainable("gating")

        vprint(self.verbose, "\n----------- Model fitted!\n\n")

    def predict(self, x: DataFrame):
        gating_vector = self.predict_gating_weights(x)
        alarm_vector = self.predict_alarm_probabilities(x)
        # predictions = alarm_vector.min(axis=0) # use this to test model without gating

        # stack the virtual decision vector on top of the alarm verdicts vector
        alarm_vector = np.vstack([np.ones((1, x.shape[0])), alarm_vector]).transpose()

        # compute final weighted average anomaly score
        predictions = np.multiply(gating_vector, alarm_vector).sum(axis=1)
        return predictions

    def predict_gating_weights(self, x):
        return self.input_to_gating.predict(x)  # .reshape((-1, self.number_of_decoders + 1))

    def predict_alarm_probabilities(self, x: DataFrame):
        alarm_vector = self.input_to_alarm.predict(x)
        alarm_vector = np.stack(alarm_vector).reshape((self.number_of_decoders, -1))
        return alarm_vector

    def predict_plot_anomalies(
        self,
        x,
        true_partitions: Optional[List[int]] = None,
        window_length: Optional[Union[int, List[int]]] = None,
        samples_per_hour: Optional[int] = 40,
        **kwargs,
    ):
        df_preds = pd.DataFrame(self.predict(x), columns=["Anomaly probability"])
        if x.index is not None:
            df_preds.index = x.index

        if window_length is not None:
            window_length = [window_length] if type(window_length) != list else window_length
            fig, ax = plt.subplots(1, 1)
            ax.plot(df_preds, label="Raw anomaly probability", alpha=0.5)
            ax.xaxis.set_major_locator(ticker.LinearLocator(numticks=8))
            plt.xticks(rotation=15)

            for window in window_length:
                legend_time_string = (
                    f"{window / samples_per_hour:0.0f} hour" if samples_per_hour is not None else f"{window} sample"
                )
                df_MA = df_preds.rolling(window=window).mean()
                col_name = str(legend_time_string + " MA")
                ax.plot(df_MA, label=col_name)
            plt.legend()
            return fig, ax

        if true_partitions:
            df_preds["partition"] = true_partitions

        if isinstance(df_preds.index[0], datetime.date):
            df_preds.index = pd.to_datetime(df_preds.index)
            df_preds.index = df_preds.index.map(lambda t: t.strftime("%d-%m-%Y"))
        fig = df_preds.plot(subplots=True, rot=15, color=["#4099DA", "red"], **kwargs)
        plt.xlabel("")
        plt.suptitle("ARGUE anomaly predictions")
        return fig

    def predict_reconstructions(self, x: DataFrame):
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
        reconstructions = np.stack(self.input_to_decoders.predict(x), axis=0)
        final_reconstructions = reconstructions[best_decoder, row_number, :]
        final_reconstructions = pd.DataFrame(final_reconstructions, columns=x.columns, index=x.index)
        return final_reconstructions

    def predict_plot_reconstructions(self, x: DataFrame, cols_to_plot: Optional[List[str]] = None, **kwargs):
        df_predictions = self.predict_reconstructions(x)
        col_names = x.columns
        col_names_pred = col_names + "_pred"
        df_predictions.columns = col_names_pred
        df_all = pd.concat([x, df_predictions], axis=1)

        swapped_col_order = []
        for i in range(len(col_names)):
            swapped_col_order.append(col_names[i])
            swapped_col_order.append(col_names_pred[i])

        df_all = df_all[swapped_col_order]
        if cols_to_plot is None:
            N_cols_to_plot = len(col_names) if len(col_names) <= 6 else 6
            cols_to_plot = df_all.columns.values[0 : 2 * N_cols_to_plot]

        df_plots = df_all[cols_to_plot]

        graphs_in_same_plot = len(col_names) == len(col_names_pred)
        if graphs_in_same_plot:
            num_plots = int(df_plots.shape[1] / 2)
            fig, axes = plt.subplots(num_plots, 1, sharex=True)
            for axis, col in zip(np.arange(num_plots), np.arange(0, df_plots.shape[1], 2)):
                df_to_plot = df_plots.iloc[:, col : col + 2]
                df_to_plot.columns = ["Actual", "Predicted"]
                if isinstance(df_to_plot.index[0], datetime.date):
                    df_to_plot.index = pd.to_datetime(df_to_plot.index)
                    df_to_plot.index = df_to_plot.index.map(lambda t: t.strftime("%d-%m-%Y"))
                df_to_plot.plot(ax=axes[axis], rot=15, legend=False)
                axes[axis].set_title(df_plots.columns[col], size=10)
                axes[axis].get_xaxis().get_label().set_visible(False)
            box = axes[axis].get_position()
            axes[axis].set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
            plt.legend(
                bbox_to_anchor=(0.7, -0.01),
                loc="lower right",
                bbox_transform=fig.transFigure,
                ncol=2,
                fancybox=True,
                shadow=True,
            )
            plt.suptitle("Model reconstructions")
            fig.tight_layout()
        else:
            for key, value in kwargs.items():
                df_all[key] = kwargs[key]
                cols_to_plot.append(key)
            fig = df_all[cols_to_plot].plot(subplots=True)
        return fig
