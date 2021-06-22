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


class BaselineAutoencoder:
    """
    BaselineAutoencoder is a baseline autoencoder model to benchmark ARGUE model types against
    """

    def __init__(self,
                 input_dim: int = 3,
                 latent_dim: int = 2,
                 residual_function: tf.keras.losses.Loss = tf.keras.losses.MeanAbsoluteError,
                 test_set_quantile_for_threshold: float = 0.995,
                 verbose: int = 1,
                 ):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = None
        self.decoder = None
        self.input_to_decoders = None
        self.history = None
        self.residual_function = residual_function(reduction=tf.keras.losses.Reduction.NONE)
        self.anomaly_threshold = None
        self.test_set_quantile_for_threshold = test_set_quantile_for_threshold
        self.verbose = verbose

    def _connect_autoencoder_pair(self, decoder):
        inputs = self.encoder.keras_model.input
        x = self.encoder.keras_model(inputs)
        outputs = decoder.keras_model(x)
        return Model(inputs, outputs, name=f"autoencoder")

    @staticmethod
    def _init_optimizer(optimizer: str,
                        learning_rate: float = 0.0003):
        optimizer = tf.keras.optimizers.get(optimizer)
        optimizer.__init__(learning_rate=learning_rate)
        return optimizer

    def _compute_anomaly_threshold(self, x_val):
        predictions = self.input_to_decoders.predict(x_val)
        residual = self.residual_function(x_val, predictions).numpy()
        return np.quantile(residual, self.test_set_quantile_for_threshold)

    def build_model(self,
                    encoder_hidden_layers: List[int] = [10, 8, 5],
                    decoders_hidden_layers: List[int] = [5, 8, 10],
                    encoder_activation: str = "tanh",
                    decoders_activation: str = "tanh",
                    all_activations: Optional[str] = None,
                    encoder_dropout_frac: Optional[float] = None,
                    decoders_dropout_frac: Optional[float] = None,
                    make_model_visualiations: bool = False,
                    autoencoder_l1: Optional[float] = None,
                    autoencoder_l2: Optional[float] = None):

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
                                                           keep_output_layer_activations=False,
                                                           l1_weight_penalty=autoencoder_l1,
                                                           l2_weight_penalty=autoencoder_l2)

        self.decoder = Network(name="decoder").build_model(input_layer=Input(shape=(self.latent_dim,)),
                                                           output_layer=Dense(self.input_dim, "sigmoid"),
                                                           units_in_layers=decoders_hidden_layers,
                                                           activation=decoders_activation,
                                                           dropout_frac=decoders_dropout_frac,
                                                           keep_output_layer_activations=False,
                                                           l1_weight_penalty=autoencoder_l1,
                                                           l2_weight_penalty=autoencoder_l2)

        decoder_output_tensor = self._connect_autoencoder_pair(self.decoder).output
        self.input_to_decoders = Model(inputs=self.encoder.keras_model.input,
                                       outputs=decoder_output_tensor)

        if make_model_visualiations:
            # if plot_model doesn't work, first pip install pydot, then pip install pydotplus, then go to:
            # https://graphviz.gitlab.io/download/ and download and install Graphviz. It must be added to
            # PATH environment variable in order to work since keras tries to call dot.exe. So
            # Graphviz\bin\ must be on the PATH.
            figures_path = get_ARGUE_figures_path()
            tf.keras.utils.plot_model(self.encoder.keras_model, to_file=figures_path / "encoder.png",
                                      show_shapes=True, show_layer_names=True)
            tf.keras.utils.plot_model(self.decoder.keras_model, to_file=figures_path / "decoder.png",
                                      show_shapes=True, show_layer_names=True)
            tf.keras.utils.plot_model(self.input_to_decoders, to_file=figures_path / "input_to_decoders.png",
                                      show_shapes=True, show_layer_names=True)

        vprint(self.verbose, f"\nARGUE networks built succesfully - properties: \n"
                             f"  > Input dimension: {self.input_dim}\n"
                             f"  > Encoder hidden layers: {encoder_hidden_layers}\n"
                             f"  > Decoder hidden layers: {decoders_hidden_layers}\n"
                             f"  > Latent dimension: {self.latent_dim}\n")
        return self

    def fit(self,
            x: Union[DataFrame, np.ndarray],
            validation_split: float = 0.1,
            batch_size: Optional[int] = 128,
            epochs: Optional[int] = 100,
            learning_rate: Union[float, List[float]] = 0.0001,
            optimizer: Union[tf.keras.optimizers.Optimizer, str] = "adam",
            stop_early: bool = False,
            stop_early_patience: int = 12,
            reduce_lr_on_plateau: bool = False,
            reduce_lr_by_factor: float = 0.5,
            reduce_lr_patience: int = 10,
            noise_factor: float = 0.0,):

        start = time.time()

        # form initial training data making sure labels and partitions are right
        vprint(self.verbose, "Preparing data: slicing into partitions and batches...\n"
                             f"Data dimensions: {x.shape}")

        autoencoder_train_dataset, autoencoder_val_dataset = train_test_split(x, test_size=validation_split)
        autoencoder_optimizer = self._init_optimizer(optimizer, learning_rate)
        autoencoder_model = self.input_to_decoders

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

        # in case we want a denoising autoencoder (noise_factor > 0) - if not data will remain unchanged
        autoencoder_train_dataset_noisy = autoencoder_train_dataset + noise_factor * np.random.normal(loc=0.0,
                                                                                                      scale=1.0,
                                                                                                      size=autoencoder_train_dataset.shape)
        autoencoder_val_dataset_noisy = autoencoder_val_dataset + noise_factor * np.random.normal(loc=0.0,
                                                                                                  scale=1.0,
                                                                                                  size=autoencoder_val_dataset.shape)

        # train autoencoder
        vprint(self.verbose, "\n\n=== Phase 1: training autoencoder ===")
        autoencoder_model.trainable = True
        autoencoder_model.compile(optimizer=autoencoder_optimizer, loss="binary_crossentropy", metrics=["MAE"])
        autoencoder_model.fit(x=autoencoder_train_dataset_noisy, y=autoencoder_train_dataset,
                              validation_data=(autoencoder_val_dataset_noisy, autoencoder_val_dataset),
                              batch_size=batch_size,
                              epochs=epochs,
                              callbacks=callbacks)
        autoencoder_model.trainable = False

        self.anomaly_threshold = self._compute_anomaly_threshold(autoencoder_val_dataset_noisy)

        end = time.time()
        time_elapsed_string = make_time_elapsed_string(end - start, 180)
        print(f"\n----------- Model fitted after:", time_elapsed_string, "\n\n")

    def predict(self, x: DataFrame):
        predictions = self.input_to_decoders.predict(x)
        anomalies = self._compute_anomalies_from_threshold(x, predictions)
        return anomalies

    def _compute_anomalies_from_threshold(self, x, predictions):
        residuals = self.residual_function(x, predictions).numpy()
        df_residuals = pd.DataFrame({"residual": residuals}, index=x.index)
        df_residuals["anomaly"] = 1*(df_residuals["residual"] >= self.anomaly_threshold)
        return df_residuals["anomaly"]

    def predict_plot_anomalies(self,
                               x,
                               window_length: Optional[Union[int, List[int]]] = None,
                               samples_per_hour: int = 40,
                               **kwargs):
        df_preds = pd.DataFrame(self.predict(x))
        if x.index is not None:
            df_preds.index = x.index

        if window_length is not None:
            window_length = [window_length] if type(window_length) != list else window_length
            fig, ax = plt.subplots(1, 1)
            ax.plot(df_preds, label="Anomaly probability", alpha=0.6)
            ax.xaxis.set_major_locator(ticker.LinearLocator(numticks=8))
            plt.xticks(rotation=15)

            for window in window_length:
                legend_time_string = f"{window / samples_per_hour:0.0f} hour" if samples_per_hour else \
                    f"{window} sample"
                df_MA = df_preds.rolling(window=window).mean()
                col_name = str(legend_time_string + " moving average")
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
        Predicts reconstructions from the autoencoder

        :param x: Dataframe with feature observations
        :return: Dataframe with reconstructed features
        """

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