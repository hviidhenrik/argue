import os
import pickle
import time
from typing import Dict, Optional, Collection

import numpy as np
import pandas as pd
from tqdm import tqdm
from pandas import DataFrame
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.python.keras.engine.functional import Functional
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.special import softmax
from sklearn.decomposition import PCA

from src.models.ARGUE.utils import *
from src.models.ARGUE.network_utils import *
from src.config.definitions import *

plt.style.use('seaborn')


class ARGUELite:
    """
    ARGUELite is a light-weight, standalone version of the more complicated ARGUE anomaly detector.
    It features
    - a single autoencoder that reduces dimension and reconstructs data
    - an alarm network that analyses hidden activation patterns from the autoencoder and
      computes anomaly probability based on these
    """

    def __init__(self,
                 input_dim: int = 3,
                 latent_dim: int = 2,
                 verbose: int = 1,
                 ):
        self.input_dim = input_dim
        self.number_of_decoders = 1
        self.latent_dim = latent_dim
        self.encoder = None
        self.decoder_dict = {}
        self.alarm = None
        self.encoder_activation_dim = None
        self.decoder_activation_dim = None
        self.input_to_decoders = None
        self.input_to_alarm = None
        self.input_to_gating = None
        self.input_to_activations = None
        self.history = None
        self.verbose = verbose

    def _connect_autoencoder_pair(self, decoder):
        decoder_number = decoder.name[8:]
        inputs = self.encoder.keras_model.input
        x = self.encoder.keras_model(inputs)
        outputs = decoder.keras_model(x)
        return Model(inputs, outputs, name=f"autoencoder_{decoder_number}")

    def _connect_encoder_input_with_activations(self, decoder,
                                                use_encoder_activations: bool = False):
        inputs = self.encoder.keras_model.input
        x = self.encoder.keras_model(inputs)
        outputs = decoder.activation_model(x)
        if use_encoder_activations:
            y = self.encoder.activation_model(inputs)
            outputs = tf.keras.layers.concatenate([outputs, y])
        return Model(inputs, outputs, name=f"input_to_activations")

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

    def _make_non_trainable(self, submodel: str = None):
        if submodel == "autoencoders":
            self.encoder.keras_model.trainable = False
            for name, network_object in self.decoder_dict.items():
                network_object.keras_model.trainable = False
        elif submodel == "alarm":
            self.alarm.keras_model.trainable = False
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

    # @tf.function
    def _alarm_gating_validation_step(self, x_batch_val, true_gating, alarm_loss, alarm_metric):
        true_alarm = 1 - true_gating[:, 1:]
        true_final = tf.reduce_min(true_alarm, axis=1)
        predicted_alarm = self.input_to_alarm(x_batch_val, training=False)

        alarm_loss_value = alarm_loss(true_alarm, predicted_alarm)
        alarm_metric.update_state(true_alarm, predicted_alarm)
        return alarm_loss_value, true_alarm, predicted_alarm

    @staticmethod
    def _init_loss_functions():
        ae_loss = tf.losses.BinaryCrossentropy()
        alarm_loss = tf.losses.BinaryCrossentropy()
        final_loss = tf.losses.BinaryCrossentropy()
        return ae_loss, alarm_loss, final_loss

    @staticmethod
    def _init_metric_functions():
        ae_metric = tf.metrics.MeanAbsoluteError()
        alarm_metric = tf.metrics.MeanAbsoluteError()
        final_metric = tf.metrics.BinaryAccuracy()  # MeanAbsoluteError()
        return ae_metric, alarm_metric, final_metric

    @staticmethod
    def _init_optimizer(optimizer: str,
                        learning_rate: float = 0.0003):
        optimizer = tf.keras.optimizers.get(optimizer)
        optimizer.__init__(learning_rate=learning_rate)
        return optimizer

    @staticmethod
    def _init_optimizer_with_lr_schedule(optimizer: str,
                                         initial_learning_rate: float = 0.0003,
                                         decay_after_epochs: int = 10,
                                         decay_rate: float = 0.7,
                                         dataset_rows: int = None,
                                         batch_size: int = None,
                                         total_epochs: int = 4,
                                         plot_schedule: bool = False,
                                         ):
        steps_per_epoch = (dataset_rows // batch_size)
        decay_steps = np.multiply(steps_per_epoch, decay_after_epochs)
        staircase = True
        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate,
                                                                       decay_steps, decay_rate,
                                                                       staircase=staircase)
        optimizer = tf.keras.optimizers.get(optimizer)
        optimizer.__init__(learning_rate=learning_rate)

        if plot_schedule:
            total_steps = total_epochs * steps_per_epoch
            plot_learning_schedule(total_steps=total_steps, initial_learning_rate=initial_learning_rate,
                                   decay_rate=decay_rate, decay_steps=decay_steps, staircase=staircase)
            plt.show()
        return optimizer

    def _unfreeze_partition_decoder(self, partition):
        self.input_to_decoders.trainable = False
        self.encoder.keras_model.trainable = True
        decoder_to_train = int(partition[10:])
        decoder_model = self.input_to_decoders.layers[decoder_to_train + 1]
        decoder_name = decoder_model.name
        if not "decoder" in decoder_name:
            raise Exception(f"Wrong model name: \"{decoder_name}\" detected in decoder training! "
                            f"Should contain \"decoder_<k>\"")
        decoder_model.trainable = True
        return decoder_model.trainable_variables

    def _init_optimizer_wrapper(self, decay_after_epochs, batch_size, epochs,
                                learning_rate, decay_rate, optimizer_object, x_train):
        if decay_after_epochs is None:
            optimizer = self._init_optimizer(optimizer=optimizer_object,
                                             learning_rate=learning_rate)
        else:
            optimizer = self._init_optimizer_with_lr_schedule(optimizer=optimizer_object,
                                                              initial_learning_rate=learning_rate,
                                                              decay_after_epochs=decay_after_epochs,
                                                              decay_rate=decay_rate,
                                                              dataset_rows=x_train.shape[0],
                                                              batch_size=batch_size,
                                                              total_epochs=epochs,
                                                              plot_schedule=True)
        return optimizer

    @staticmethod
    def _prepare_data(n_noise_samples, noise_stdev, noise_stdevs_away, plot_normal_vs_noise,
                      validation_split, x):
        x_copy = x.copy()
        x_copy["partition"] = 1
        # make gaussian noise samples so the optimization doesn't only see "healthy" data
        # and hence just learns to always predict healthy, i.e. P(healthy) = certain
        # TODO revise noise distribution or do it at runtime/training time instead (on the fly)
        x_noise = generate_noise_samples(x_copy.drop(columns=["partition"]),
                                         quantiles=[0.005, 0.995], stdev=noise_stdev,
                                         stdevs_away=noise_stdevs_away, n_noise_samples=n_noise_samples)
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
        normal_vs_noise_one_hot_vector = pd.get_dummies(x_with_noise_and_labels["partition"]).values
        x_train, x_val, one_hot_train_labels, one_hot_val_labels = train_test_split(x_with_noise_and_labels,
                                                                                    normal_vs_noise_one_hot_vector,
                                                                                    test_size=validation_split)
        x_train = x_train.reset_index(drop=True)
        x_val = x_val.reset_index(drop=True)
        alarm_train_labels = 1 - one_hot_train_labels[:, 1:]
        alarm_val_labels = 1 - one_hot_val_labels[:, 1:]
        return alarm_train_labels, alarm_val_labels, x_train, x_val

    def build_model(self,
                    encoder_hidden_layers: List[int] = [10, 8, 5],
                    decoders_hidden_layers: List[int] = [5, 8, 10],
                    alarm_hidden_layers: List[int] = [15, 12, 10],
                    encoder_activation: str = "tanh",
                    decoders_activation: str = "tanh",
                    alarm_activation: str = "tanh",
                    all_activations: Optional[str] = None,
                    use_encoder_activations_in_alarm: bool = True,
                    use_latent_activations_in_encoder_activations: bool = True,
                    use_decoder_outputs_in_decoder_activations: bool = True,
                    encoder_dropout_frac: Optional[float] = None,
                    decoders_dropout_frac: Optional[float] = None,
                    alarm_dropout_frac: Optional[float] = None,
                    make_model_visualiations: bool = False,
                    autoencoder_l1: Optional[float] = None,
                    autoencoder_l2: Optional[float] = None,
                    alarm_l1: Optional[float] = None,
                    alarm_l2: Optional[float] = None, ):

        # if all_activations is specified, the same activation function is used in all hidden layers
        if all_activations is not None:
            encoder_activation = all_activations
            decoders_activation = all_activations
            alarm_activation: all_activations

        # build shared encoder
        self.encoder = Network(name="encoder").build_model(input_layer=Input(shape=(self.input_dim,)),
                                                           output_layer=Dense(self.latent_dim, encoder_activation),
                                                           units_in_layers=encoder_hidden_layers,
                                                           activation=encoder_activation,
                                                           dropout_frac=encoder_dropout_frac,
                                                           keep_output_layer_activations=use_latent_activations_in_encoder_activations,
                                                           l1_weight_penalty=autoencoder_l1,
                                                           l2_weight_penalty=autoencoder_l2)

        # set constants
        self.encoder_activation_dim = self.encoder.get_activation_dim()
        self.decoder_activation_dim = int(np.sum(decoders_hidden_layers))
        if use_decoder_outputs_in_decoder_activations:
            self.decoder_activation_dim += self.input_dim

        # build alarm network
        alarm_input_dim = self.decoder_activation_dim
        if use_encoder_activations_in_alarm:
            alarm_input_dim += self.encoder_activation_dim
        self.alarm = Network(name="alarm").build_model(input_layer=Input(shape=(alarm_input_dim,)),
                                                       output_layer=Dense(1, "sigmoid"),
                                                       units_in_layers=alarm_hidden_layers,
                                                       activation=alarm_activation,
                                                       dropout_frac=alarm_dropout_frac,
                                                       l1_weight_penalty=alarm_l1,
                                                       l2_weight_penalty=alarm_l2)

        alarm_outputs = []
        decoder_outputs = []
        for i in range(1, self.number_of_decoders + 1):
            decoder_name = f"decoder_{i}"
            decoder = Network(name=decoder_name).build_model(input_layer=Input(shape=(self.latent_dim,)),
                                                             output_layer=Dense(self.input_dim, "sigmoid"),
                                                             units_in_layers=decoders_hidden_layers,
                                                             activation=decoders_activation,
                                                             dropout_frac=decoders_dropout_frac,
                                                             keep_output_layer_activations=use_decoder_outputs_in_decoder_activations,
                                                             l1_weight_penalty=autoencoder_l1,
                                                             l2_weight_penalty=autoencoder_l2)
            self.decoder_dict[decoder_name] = decoder
            # connect common encoder with each decoder
            decoder_output_tensor = self._connect_autoencoder_pair(decoder).output
            decoder_outputs.append(decoder_output_tensor)
            # connect encoder with alarm model through each decoder/expert network

            alarm_output_tensor = self._connect_alarm_pair(decoder, use_encoder_activations_in_alarm).output
            alarm_outputs.append(alarm_output_tensor)

        self.input_to_activations = self._connect_encoder_input_with_activations(decoder,
                                                                                 use_encoder_activations_in_alarm)
        self.input_to_decoders = Model(inputs=self.encoder.keras_model.input,
                                       outputs=decoder_outputs)

        self.input_to_decoders.trainable = False
        self.encoder.keras_model.trainable = True

        self.input_to_alarm = Model(inputs=self.encoder.keras_model.input,
                                    outputs=alarm_outputs)

        if make_model_visualiations:
            # if plot_model doesn't work, first pip install pydot, then pip install pydotplus, then go to:
            # https://graphviz.gitlab.io/download/ and download and install Graphviz. It must be added to
            # PATH environment variable in order to work since keras tries to call dot.exe. So
            # Graphviz\bin\ must be on the PATH.
            figures_path = get_ARGUE_figures_path()
            tf.keras.utils.plot_model(self.encoder.keras_model, to_file=figures_path / "encoder.png",
                                      show_shapes=True, show_layer_names=True)
            tf.keras.utils.plot_model(self.encoder.activation_model, to_file=figures_path / "encoder_activations.png",
                                      show_shapes=True, show_layer_names=True)
            tf.keras.utils.plot_model(self.decoder_dict["decoder_1"].keras_model, to_file=figures_path / "decoder.png",
                                      show_shapes=True, show_layer_names=True)
            tf.keras.utils.plot_model(self.decoder_dict["decoder_1"].activation_model,
                                      to_file=figures_path / "decoder_activations.png",
                                      show_shapes=True, show_layer_names=True)
            tf.keras.utils.plot_model(self.alarm.keras_model, to_file=figures_path / "alarm.png",
                                      show_shapes=True, show_layer_names=True)
            tf.keras.utils.plot_model(self.input_to_decoders, to_file=figures_path / "input_to_decoders.png",
                                      show_shapes=True, show_layer_names=True)
            tf.keras.utils.plot_model(self.input_to_alarm, to_file=figures_path / "input_to_alarm.png",
                                      show_shapes=True, show_layer_names=True)

        vprint(self.verbose, f"\nARGUE networks built succesfully - properties: \n"
                             f"  > Input dimension: {self.input_dim}\n"
                             f"  > Encoder activations: {self.encoder_activation_dim}\n"
                             f"  > Decoder activations: {self.decoder_activation_dim}\n"
                             f"  > Latent dimension: {self.latent_dim}\n"
                             f"  > Number of decoders: {self.number_of_decoders}\n")
        return self

    def fit(self,
            x: Union[DataFrame, np.ndarray],
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
            ae_learning_rate: Union[float, List[float]] = 0.0001,
            alarm_gating_learning_rate: float = 0.0001,
            autoencoder_decay_after_epochs: Optional[Union[int, List[int]]] = None,
            alarm_decay_after_epochs: Optional[int] = None,
            decay_rate: Optional[float] = 0.7,  # 0.1 = heavy reduction, 0.9 = slight reduction
            optimizer: Union[tf.keras.optimizers.Optimizer, str] = "adam",
            plot_normal_vs_noise: bool = False,
            stop_early: bool = False,
            stop_early_patience: int = 15,
            reduce_lr_on_plateau: bool = False,
            reduce_lr_by_factor: float = 0.5,
            reduce_lr_patience: int = 10,
            noise_factor: float = 0.0):

        start = time.time()
        autoencoder_epochs = epochs if autoencoder_epochs is None else autoencoder_epochs
        alarm_gating_epochs = epochs if alarm_gating_epochs is None else alarm_gating_epochs
        autoencoder_batch_size = batch_size if autoencoder_batch_size is None else autoencoder_batch_size
        alarm_gating_batch_size = batch_size if alarm_gating_batch_size is None else alarm_gating_batch_size

        # form initial training data making sure labels and partitions are right
        vprint(self.verbose, "Preparing data: slicing into partitions and batches...\n"
                             f"Data dimensions: {x.shape}")

        # make datasets ready
        alarm_train_labels, alarm_val_labels, x_train, x_val = self._prepare_data(n_noise_samples, noise_stdev,
                                                                                  noise_stdevs_away,
                                                                                  plot_normal_vs_noise,
                                                                                  validation_split, x)
        ae_train_dataset = x_train[x_train["partition"] == 1].drop(columns=["partition"])
        ae_val_dataset = x_val[x_val["partition"] == 1].drop(columns=["partition"])

        ae_optimizer = self._init_optimizer_wrapper(autoencoder_decay_after_epochs, autoencoder_batch_size,
                                                    autoencoder_epochs, ae_learning_rate, decay_rate,
                                                    optimizer, x_train)
        alarm_optimizer = self._init_optimizer_wrapper(alarm_decay_after_epochs, alarm_gating_batch_size,
                                                       alarm_gating_epochs, alarm_gating_learning_rate, decay_rate,
                                                       optimizer, x_train)
        ae_model = self.input_to_decoders
        alarm_model = self.alarm.keras_model

        callbacks = []
        # callbacks.append(tf.keras.callbacks.TensorBoard(log_dir="./logs", histogram_freq=5))
        if stop_early:
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=stop_early_patience,
                                                 restore_best_weights=True, mode="min"))
        if reduce_lr_on_plateau:
            callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=reduce_lr_by_factor,
                                                                  patience=reduce_lr_patience, verbose=1,
                                                                  mode="min"))

        ae_train_dataset_noisy = ae_train_dataset + noise_factor * np.random.normal(loc=0.0,
                                                                                    scale=1.0,
                                                                                    size=ae_train_dataset.shape)
        ae_val_dataset_noisy = ae_val_dataset + noise_factor * np.random.normal(loc=0.0,
                                                                                scale=1.0,
                                                                                size=ae_val_dataset.shape)

        # train autoencoder
        vprint(self.verbose, "\n\n=== Phase 1: training autoencoder ===")
        ae_model.trainable = True
        ae_model.compile(optimizer=ae_optimizer, loss="binary_crossentropy", metrics=["MAE"])
        ae_model.fit(x=ae_train_dataset_noisy, y=ae_train_dataset,
                     validation_data=(ae_val_dataset_noisy, ae_val_dataset),
                     batch_size=autoencoder_batch_size,
                     epochs=autoencoder_epochs,
                     callbacks=callbacks)
        ae_model.trainable = False

        # make alarm dataset from fully trained autoencoder activations
        alarm_train_dataset = self.input_to_activations.predict(x_train.drop(columns=["partition"]))
        alarm_val_dataset = self.input_to_activations.predict(x_val.drop(columns=["partition"]))

        # train alarm network
        vprint(self.verbose, "\n\n=== Phase 2: training alarm network ===")
        alarm_model.compile(optimizer=alarm_optimizer, loss="binary_crossentropy", metrics=["MAE"])
        alarm_model.fit(x=alarm_train_dataset, y=alarm_train_labels,
                        validation_data=(alarm_val_dataset, alarm_val_labels),
                        batch_size=alarm_gating_batch_size,
                        epochs=alarm_gating_epochs,
                        callbacks=callbacks)

        end = time.time()
        time_elapsed_string = make_time_elapsed_string(end - start, 180)
        print(f"\n----------- Model fitted after:", time_elapsed_string, "\n\n")

    def predict(self, x: DataFrame):
        return self.input_to_alarm.predict(x)

    def predict_plot_anomalies(self,
                               x,
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
        reconstructions = self.input_to_decoders.predict(x)
        final_reconstructions = pd.DataFrame(reconstructions,
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
