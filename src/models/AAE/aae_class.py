import tensorflow as tf

tf.get_logger().setLevel('ERROR')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from src.utilities.utility_functions import *
from src.data.data_utils import *
from src.config.definitions import *
import time

plt.style.use("seaborn")

MODEL_NUMBER_TO_NAME_MAPPING = {"0": "encoder", "1": "decoder", "2": "discriminator"}


# TODO add validation feature to training
# TODO add early stopping feature
# TODO add weight clipping
# TODO add batch normalization
class AdversarialAutoencoder:
    def __init__(self,
                 latent_layer_dimension: int = 2,
                 encoder_hidden_layers: List = None,
                 decoder_hidden_layers: List = None,
                 discrim_hidden_layers: List = None,
                 autoencoder_activation_function: tf.keras.layers = "elu",
                 discrim_activation_function: tf.keras.layers = "elu",
                 batch_normalization: bool = True,
                 ):
        self.latent_layer_dimension = latent_layer_dimension
        self.encoder_hidden_layers = encoder_hidden_layers if encoder_hidden_layers is not None else [10, 8, 6, 4]
        self.decoder_hidden_layers = decoder_hidden_layers if decoder_hidden_layers is not None else [4, 6, 8, 10]
        self.discrim_hidden_layers = discrim_hidden_layers if discrim_hidden_layers is not None else [10, 8, 6, 4]
        self.autoencoder_activation_function = autoencoder_activation_function
        self.discrim_activation_function = discrim_activation_function
        self.batch_normalization = batch_normalization
        self.discriminator = None
        self.encoder = None
        self.decoder = None
        self.epochs = None
        self.batch_size = None
        self.number_of_features = None
        self.number_of_train_data_rows = None
        self.history = {"epochs": [], "ae_loss": [], "discrim_loss": [], "generator_loss": []}
        self._autoencoder_optimizer = None
        self._discriminator_optimizer = None
        self._generator_optimizer = None
        self._autoencoder_epoch_mean_loss = tf.metrics.Mean()
        self._discriminator_epoch_mean_loss = tf.metrics.Mean()
        self._discriminator_epoch_mean_accuracy = tf.metrics.Mean()
        self._generator_epoch_mean_loss = tf.metrics.Mean()

    def _build_encoder_model(self, dropout_fraction: float = 0.2):
        hidden_layers = self.encoder_hidden_layers
        inputs = tf.keras.Input(shape=(self.number_of_features,))
        x = tf.keras.layers.Dense(units=hidden_layers[0], activation=self.autoencoder_activation_function)(inputs)
        x = tf.keras.layers.Dropout(dropout_fraction)(x)
        if self.batch_normalization:
            x = tf.keras.layers.BatchNormalization()(x)
        remaining_layers = hidden_layers[1:]
        for units in remaining_layers:
            x = tf.keras.layers.Dense(units=units, activation=self.autoencoder_activation_function)(x)
            x = tf.keras.layers.Dropout(dropout_fraction)(x)
            if self.batch_normalization:
                x = tf.keras.layers.BatchNormalization()(x)
        latent_space = tf.keras.layers.Dense(self.latent_layer_dimension, activation="linear")(x)
        model = tf.keras.Model(inputs=inputs, outputs=latent_space)
        self.encoder = model

    def _build_decoder_model(self, dropout_fraction: float = 0.2):
        hidden_layers = self.decoder_hidden_layers
        latent_space = tf.keras.Input(shape=(self.latent_layer_dimension,))
        x = tf.keras.layers.Dense(units=hidden_layers[0],
                                  activation=self.autoencoder_activation_function)(latent_space)
        x = tf.keras.layers.Dropout(dropout_fraction)(x)
        if self.batch_normalization:
            x = tf.keras.layers.BatchNormalization()(x)
        remaining_layers = hidden_layers[1:]
        for units in remaining_layers:
            x = tf.keras.layers.Dense(units=units, activation=self.autoencoder_activation_function)(x)
            x = tf.keras.layers.Dropout(dropout_fraction)(x)
            if self.batch_normalization:
                x = tf.keras.layers.BatchNormalization()(x)
        reconstruction = tf.keras.layers.Dense(self.number_of_features, activation="linear")(x)
        model = tf.keras.Model(inputs=latent_space, outputs=reconstruction)
        self.decoder = model

    def _build_discriminator_model(self, dropout_fraction: float = 0.2):
        hidden_layers = self.discrim_hidden_layers
        latent_space = tf.keras.Input(shape=(self.latent_layer_dimension,))
        x = tf.keras.layers.Dense(units=hidden_layers[0],
                                  activation=self.discrim_activation_function)(latent_space)
        x = tf.keras.layers.Dropout(dropout_fraction)(x)
        # if self.batch_normalization:
        #     x = tf.keras.layers.BatchNormalization()(x)
        remaining_layers = hidden_layers[1:]
        for units in remaining_layers:
            x = tf.keras.layers.Dense(units=units, activation=self.discrim_activation_function)(x)
            x = tf.keras.layers.Dropout(dropout_fraction)(x)
            if self.batch_normalization:
                x = tf.keras.layers.BatchNormalization()(x)
        prediction_real_or_fake = tf.keras.layers.Dense(1, activation="linear")(x)
        model = tf.keras.Model(inputs=latent_space, outputs=prediction_real_or_fake)
        self.discriminator = model

    def _set_epoch_mean_loss(self,
                             autoencoder_loss,
                             discriminator_loss,
                             discriminator_accuracy,
                             generator_loss,
                             epoch
                             ):
        self._autoencoder_epoch_mean_loss(autoencoder_loss)
        self._discriminator_epoch_mean_loss(discriminator_loss)
        self._discriminator_epoch_mean_accuracy(discriminator_accuracy)
        self._generator_epoch_mean_loss(generator_loss)

        self.history["epochs"].append(epoch)
        self.history["ae_loss"].append(self._autoencoder_epoch_mean_loss.result())
        self.history["discrim_loss"].append(self._discriminator_epoch_mean_loss.result())
        self.history["generator_loss"].append(self._generator_epoch_mean_loss.result())

    def _get_losses(self):
        """
        Get current mean losses from all the models
        :return: last recorded mean loss for all the models,
        :rtype: tuple
        """
        return (self._autoencoder_epoch_mean_loss.result(),
                self._discriminator_epoch_mean_loss.result(),
                self._discriminator_epoch_mean_accuracy.result(),
                self._generator_epoch_mean_loss.result())

    @staticmethod
    def _reconstruction_loss(inputs, reconstruction, loss_weight: float = 1):
        loss = tf.keras.losses.MeanSquaredError()
        return loss_weight * loss(inputs, reconstruction)

    @staticmethod
    def _discriminator_loss(real_output, fake_output, loss_weight: float = 1):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        loss_real = cross_entropy(tf.ones_like(real_output), real_output)
        loss_fake = cross_entropy(tf.zeros_like(fake_output), fake_output)
        loss = loss_real + loss_fake
        loss = tf.reduce_mean(real_output) - tf.reduce_mean(fake_output)
        return loss_weight * loss

    @staticmethod
    def _generator_loss(fake_output, loss_weight: float = 1):
        """
        The binary crossentropy is calculated here from the result of sending the generated code through
        the discriminator network. The output from the discriminator is 1 (real) if it
        is perfectly fooled by the generator, otherwise it should be 0 (not real) if it's not fooled at all.
        Therefore, the generator/encoder will be optimized to produce latent codes that will fool the
        discriminator to think it is actually a sample drawn from the chosen prior distribution.
        """
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        # loss = cross_entropy(tf.ones_like(fake_output), fake_output)
        loss = fake_output
        return loss_weight * loss

    def _prepare_dataset_in_batches(self, df_features: DataFrame):
        train_dataset = tf.data.Dataset.from_tensor_slices(df_features)
        train_dataset = train_dataset.shuffle(buffer_size=self.number_of_train_data_rows)
        train_dataset = train_dataset.batch(self.batch_size)
        return train_dataset

    @staticmethod
    def _get_updated_learning_rate(base_learning_rate, global_step, max_learning_rate, step_size):
        global_step = global_step + 1
        cycle = np.floor(1 + global_step / (2 * step_size))
        x_lr = np.abs(global_step / step_size - 2 * cycle + 1)
        new_learning_rate = base_learning_rate + (max_learning_rate - base_learning_rate) * max(0, 1 - x_lr)
        return new_learning_rate

    def _set_learning_rates(self, learning_rate):
        self._autoencoder_optimizer.lr = learning_rate
        self._discriminator_optimizer.lr = learning_rate
        self._generator_optimizer.lr = learning_rate

    def _compute_losses_and_update_weights(self, batch_x):
        def _concat_tf_variables(tf_array_1, tf_array_2):
            return tf.concat([tf_array_1, tf_array_2], axis=0)

        def _compute_discriminator_accuracy(discriminator_real_sample_output,
                                            discriminator_fake_sample_output):
            accuracy = tf.keras.metrics.BinaryAccuracy()
            discriminator_true_labels = _concat_tf_variables(tf.ones_like(discriminator_real_sample_output),
                                                             tf.zeros_like(discriminator_fake_sample_output))
            discriminator_predicted_labels = _concat_tf_variables(discriminator_real_sample_output,
                                                                  discriminator_fake_sample_output)
            return accuracy(discriminator_true_labels, discriminator_predicted_labels)

        # -------------------------------------------------------------------------------------------------------------
        # Autoencoder
        with tf.GradientTape() as autoencoder_tape:
            encoder_output = self.encoder(batch_x, training=True)
            decoder_output = self.decoder(encoder_output, training=True)
            reconstruction_loss = self._reconstruction_loss(batch_x, decoder_output, loss_weight=1)

        autoencoder_trainable_variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        autoencoder_gradients = autoencoder_tape.gradient(reconstruction_loss, autoencoder_trainable_variables)
        self._autoencoder_optimizer.apply_gradients(zip(autoencoder_gradients, autoencoder_trainable_variables))

        # -------------------------------------------------------------------------------------------------------------
        # Discriminator
        with tf.GradientTape() as discriminator_tape:
            real_distribution = tf.random.normal([batch_x.shape[0], self.latent_layer_dimension],
                                                 mean=0, stddev=0.01)
            encoder_output = self.encoder(batch_x, training=True)

            discriminator_real_sample_output = self.discriminator(real_distribution, training=True)
            discriminator_fake_sample_output = self.discriminator(encoder_output, training=True)

            discriminator_loss = self._discriminator_loss(discriminator_real_sample_output,
                                                          discriminator_fake_sample_output, loss_weight=1)

            discriminator_accuracy = _compute_discriminator_accuracy(discriminator_real_sample_output,
                                                                     discriminator_fake_sample_output)

        discriminator_gradients = discriminator_tape.gradient(discriminator_loss,
                                                              self.discriminator.trainable_variables)
        self._discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                          self.discriminator.trainable_variables))
        # -------------------------------------------------------------------------------------------------------------
        # Generator (Encoder)
        with tf.GradientTape() as generator_tape:
            encoder_output = self.encoder(batch_x, training=True)
            discriminator_fake_sample_output = self.discriminator(encoder_output, training=True)

            generator_loss = self._generator_loss(discriminator_fake_sample_output)

        generator_gradients = generator_tape.gradient(generator_loss, self.encoder.trainable_variables)
        self._generator_optimizer.apply_gradients(zip(generator_gradients,
                                                      self.encoder.trainable_variables))
        return reconstruction_loss, discriminator_loss, discriminator_accuracy, generator_loss

    def fit(self,
            df_features_scaled: DataFrame,
            batch_size: int = 128,
            epochs: int = 10,
            dropout_fraction: float = 0.2,
            base_learning_rate: float = 0.00025,
            max_learning_rate: float = 0.0025,
            verbose: int = 2):

        self.number_of_train_data_rows = df_features_scaled.shape[0]
        self.number_of_features = df_features_scaled.shape[1]
        self.batch_size = batch_size
        self.epochs = epochs
        self._build_encoder_model(dropout_fraction=dropout_fraction)
        self._build_decoder_model(dropout_fraction=dropout_fraction)
        self._build_discriminator_model(dropout_fraction=dropout_fraction)
        self._autoencoder_optimizer = tf.keras.optimizers.Adam(learning_rate=base_learning_rate)
        self._discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=base_learning_rate)
        self._generator_optimizer = tf.keras.optimizers.Adam(learning_rate=base_learning_rate)

        step_size = 2 * np.ceil(self.number_of_train_data_rows / self.batch_size)
        global_step = 0
        train_dataset_in_batches = self._prepare_dataset_in_batches(df_features_scaled)
        # Training loop
        for epoch in range(self.epochs):
            start = time.time()

            # Learning rate schedule
            if epoch in [60, 100, 300]:
                base_learning_rate = base_learning_rate / 2
                max_learning_rate = max_learning_rate / 2
                step_size = step_size / 2
                print('Learning rate decreased!')

            for batch_number, batch in enumerate(train_dataset_in_batches):
                if verbose > 2:
                    number_of_batches = int(np.floor(self.number_of_train_data_rows / self.batch_size))
                    print(f"Epoch {epoch + 1}, batch {batch_number + 1}/{number_of_batches + 1}")
                new_learning_rate = self._get_updated_learning_rate(base_learning_rate,
                                                                    global_step,
                                                                    max_learning_rate,
                                                                    step_size)
                self._set_learning_rates(new_learning_rate)
                losses = self._compute_losses_and_update_weights(batch)
            self._set_epoch_mean_loss(*losses, epoch)

            epoch_time = time.time() - start
            if verbose > 1:
                print(
                    f'=======> Epoch {epoch + 1}\nTime: {epoch_time:.2f}s, '
                    f'ETA: {epoch_time * (self.epochs - epoch):.0f}s'
                )
                print(
                    'AE loss: {:.4f}, discriminator loss: {:.4f}, '
                    'discriminator accuracy: {:.4f}, '
                    'generator loss: {:.4f}'.format(*self._get_losses())
                )
        if verbose > 0:
            print("\n... Fitting procedure complete - model fitted succesfully!")

    def predict(self, df_features_scaled):
        latent_space = self.encoder.predict(df_features_scaled)
        return self.decoder.predict(latent_space)

    def predict_latent_space(self, df_features_scaled):
        return self.encoder.predict(df_features_scaled)

    def plot_latent_space(self, df_features_scaled, coloring_column: DataFrame = None):
        df_latent_space = pd.DataFrame(self.predict_latent_space(df_features_scaled))
        title_latent = "AAE latent space"
        pca_plot = self.latent_layer_dimension > 2
        if pca_plot:
            pca = PCA(n_components=2)
            df_latent_space = pd.DataFrame(pca.fit_transform(df_latent_space))
            var_expl = 100 * pca.explained_variance_ratio_.sum()
            title_latent = title_latent + f"\nPCA transformed (variance explained:{var_expl:4.0f}%)"
        if coloring_column is None:
            color = pd.DataFrame(df_features_scaled).iloc[:, 0]
            colorbar_label = "First (scaled) feature column"
        else:
            color = coloring_column.iloc[:, 0]
            colorbar_label = coloring_column.columns.values[0]

        fig = plt.scatter(df_latent_space.iloc[:, 0], df_latent_space.iloc[:, 1],
                          c=color, cmap='jet', s=10)
        plt.xlabel("PC1" if pca_plot else "z0")
        plt.ylabel("PC2" if pca_plot else "z1")
        clb = plt.colorbar()
        clb.set_label(colorbar_label, rotation=0, labelpad=-30, y=1.05)
        plt.title(title_latent)
        return fig

    def plot_learning(self):
        """
        Plots learning curve given a history object from a trained Keras model

        :param history: history object from a Keras model
        """
        fig = plt.plot(self.history["epochs"], self.history["ae_loss"], label="autoencoder")
        plt.plot(self.history["epochs"], self.history["discrim_loss"], label="discriminator")
        plt.plot(self.history["epochs"], self.history["generator_loss"], label="generator")
        # if len(history.keys()):
        #     plt.plot(history["val_loss"])
        #     plt.legend(["train", "validation"], loc="upper right")
        # else:
        #     plt.legend(["train"], loc="upper right")
        plt.title("Learning curve")
        plt.legend()
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        return fig

    @staticmethod
    def plot_predictions(
            df_input: DataFrame,
            df_predictions: DataFrame,
            cols_to_plot: List[str] = None,
            **kwargs,
    ):
        """
        Plots data and its autoencoder based reconstructions vs each other along
        with the mean squared error (mse) for visual comparison. Additional columns to be plotted can be added
        via keyword arguments (kwargs).

        :param df_input: the original data used as input to the autoencoder
        :type df_input: pandas.core.frame.DataFrame
        :param df_predictions: the autoencoder reconstructions of the original data in df_input
        :type df_predictions: pandas.core.frame.DataFrame
        :param cols_to_plot: the columns from df_input to plot, use "column_name_pred" for corresponding predictions
        :type cols_to_plot: List[str], default=None
        :param lstm_model: is the model a neural network of LSTM (recurrent time based network) type?
        :type lstm_model: bool, default=False
        """

        col_names = df_input.columns
        col_names_pred = col_names + "_pred"
        df_predictions.columns = col_names_pred
        df_all = pd.concat([df_input, df_predictions], 1)

        swapped_col_order = []
        for i in range(len(col_names)):
            swapped_col_order.append(col_names[i])
            swapped_col_order.append(col_names_pred[i])

        df_all = df_all[swapped_col_order]
        if cols_to_plot is None:
            N_cols_to_plot = len(col_names) if len(col_names) <= 5 else 5
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

    def save(self, filename: str = None):
        model_list = [self.encoder, self.decoder, self.discriminator]
        for model_number, model in enumerate(model_list):
            model_name = MODEL_NUMBER_TO_NAME_MAPPING[str(model_number)]
            if filename is None:
                filename = get_model_archive_path() / model_name
            else:
                filename = get_model_archive_path() / f"{filename}_{model_name}"
            tf.keras.models.save_model(model, filename)
        print(f"Model saved succesfully in {get_model_archive_path()}")

    def load(self, filename: str = None):
        model_list = [self.encoder, self.decoder, self.discriminator]
        loaded_models = {"encoder": None, "decoder": None, "discriminator": None}
        for model_number, loaded_model in enumerate(model_list):
            model = MODEL_NUMBER_TO_NAME_MAPPING[str(model_number)]
            if filename is None:
                filename = get_model_archive_path() / model
            else:
                filename = get_model_archive_path() / f"{filename}_{model}"
            loaded_models[model] = tf.keras.models.load_model(filename, compile=False)
        self.encoder = loaded_models["encoder"]
        self.decoder = loaded_models["decoder"]
        self.discriminator = loaded_models["discriminator"]
        print(f"Model loaded succesfully from {get_model_archive_path()}")


if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    is_debugging = True
    # is_debugging = False
    pd.set_option('display.max_columns', None)
    filename = get_pump_data_path() / f"data_cwp_pump_10_{get_dataset_purpose_as_str(is_debugging)}.csv"
    df_features = get_local_data(filename)
    df_features = df_features.dropna()
    N_rows = 10000
    # df_features = pd.DataFrame(
    #     {
    #         "x1": np.sin(np.linspace(0, 10, N_rows) + np.random.normal(0, 0.1, N_rows)),
    #         "x2": np.cos(np.linspace(0, 10, N_rows) + np.random.normal(0, 0.1, N_rows)),
    #         "x3": np.cos(3.14 + np.linspace(0, 10, N_rows) + np.random.normal(0, 0.1, N_rows)),
    #         "x4": np.cos(2 - np.exp(np.linspace(0, 10, N_rows)) + np.random.normal(0, 0.1, N_rows))
    #     }
    # )
    print(df_features.shape)

    aae = AdversarialAutoencoder(latent_layer_dimension=2,
                                 encoder_hidden_layers=[20, 40, 20, 16, 10],
                                 decoder_hidden_layers=[10, 16, 20, 40, 20],
                                 discrim_hidden_layers=[3, 4, 4, 3],
                                 batch_normalization=False)
    scaler = MinMaxScaler()
    df_features_scaled = scaler.fit_transform(df_features)
    aae.fit(df_features_scaled[0:N_rows,], dropout_fraction=0, epochs=100, batch_size=512, verbose=2)
    aae.save("test_aae")

    aae.plot_learning()
    plt.show()
    # aae.plot_latent_space(df_features_scaled, df_features[["x1"]])
    # plt.show()
    # aae.plot_latent_space(df_features_scaled, df_features[["x2"]])
    # plt.show()
    # aae.plot_latent_space(df_features_scaled, df_features[["x3"]])
    # plt.show()
    # aae.plot_latent_space(df_features_scaled, df_features[["x4"]])
    # plt.show()
    aae.plot_latent_space(df_features_scaled[0:N_rows,])
    plt.show()
    # aae.plot_latent_space(df_features_scaled[0:N_rows,], df_features.iloc[0:N_rows,1])
    # plt.show()
    # aae.plot_latent_space(df_features_scaled[0:N_rows,], df_features.iloc[0:N_rows,2])
    # plt.show()

    df_to_predict = pd.DataFrame(df_features_scaled[N_rows: N_rows + 200, :], columns=df_features.columns)
    df_predictions = pd.DataFrame(aae.predict(df_to_predict), columns=df_features.columns)

    aae.plot_predictions(df_to_predict, df_predictions)
    plt.show()
