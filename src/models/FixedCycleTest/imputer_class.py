from typing import Union, List
import tensorflow as tf
import pandas as pd
import pickle
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from src.config.definitions import *


class FFNImputer:
    def __init__(self,
                 columns_to_impute: List[str],
                 hidden_layers: List[int] = [8, 5, 4],
                 activation_functions: Union[str, List[str]] = "elu",
                 epochs: int = 30,
                 loss: str = "binary_crossentropy",
                 metrics: str = "accuracy"):
        self.columns_to_impute = columns_to_impute
        self.hidden_layers = hidden_layers
        self.activation_functions = activation_functions
        self.epochs = epochs
        self.loss = loss
        self.metrics = metrics
        self.scaler = MinMaxScaler()
        self.model = None

    def _build_imputer_network(self, df: DataFrame, columns_to_impute: List[str]):
        """
        Builds the autoencoder based on the parameters given when instantiated and the call to fit

        :return:
        """
        N_features = df.shape[1] - len(columns_to_impute)
        N_targets = len(columns_to_impute)
        hidden_layers = self.hidden_layers
        if isinstance(self.activation_functions, str):
            activation_functions = [
                self.activation_functions for layer in hidden_layers
            ]
            last_activation = "sigmoid" if self.loss == "binary_crossentropy" else "linear"
            activation_functions.append(last_activation)
        else:
            assert len(self.activation_functions) == len(self.hidden_layers) + 1, (
                "Incorrect number of activation functions provided. The number must be |hidden_layers| + 1. "
                "Did you forget the output layer function?"
            )
            activation_functions = self.activation_functions

        inputs = tf.keras.layers.Input(shape=(N_features,))
        x = tf.keras.layers.Dense(units=hidden_layers[0], activation=activation_functions[0])(inputs)
        x = tf.keras.layers.Dropout(0)(x)
        remaining_layers, remaining_activation_functions = (
            hidden_layers[1:],
            activation_functions[1:],
        )
        for units, activation in zip(remaining_layers, remaining_activation_functions):
            x = tf.keras.layers.Dense(units=units, activation=activation)(x)
            x = tf.keras.layers.Dropout(0)(x)
        outputs = tf.keras.layers.Dense(N_targets, activation=activation_functions[-1])(x)
        imputer = tf.keras.Model(inputs=inputs, outputs=outputs)
        imputer.compile(optimizer="adam", loss=self.loss, metrics=self.metrics)
        self.model = imputer

    def fit(self, df: DataFrame, **kwargs) -> None:
        df_copy = df.copy()
        df_copy = df_copy.dropna()

        df_train, df_test = train_test_split(df_copy.dropna(), train_size=0.85)

        df_train_scaled = pd.DataFrame(self.scaler.fit_transform(df_train),
                                       columns=df_train.columns,
                                       index=df_train.index)
        df_test_scaled = pd.DataFrame(self.scaler.transform(df_test),
                                      columns=df_test.columns,
                                      index=df_test.index)

        self._build_imputer_network(df_train_scaled, columns_to_impute=self.columns_to_impute)
        self.model.fit(x=df_train_scaled.drop(columns=self.columns_to_impute),
                       y=df_train_scaled[self.columns_to_impute],
                       epochs=self.epochs,
                       validation_split=0.15,
                       **kwargs)
        print("Model evaluation:\n", self.model.evaluate(df_test_scaled.drop(columns=self.columns_to_impute),
                                  df_test_scaled[self.columns_to_impute]))

    def impute(self, df_with_na: DataFrame) -> DataFrame:
        df_only_na = df_with_na[df_with_na[self.columns_to_impute].isna().any(axis=1)]
        df_without_na = df_with_na[~df_with_na[self.columns_to_impute].isna().any(axis=1)]
        df_only_na_scaled = pd.DataFrame(self.scaler.transform(df_only_na),
                                         columns=df_only_na.columns,
                                         index=df_only_na.index)
        df_only_na_scaled_without_impute_columns = df_only_na_scaled.drop(columns=self.columns_to_impute)
        df_imputations = self.model.predict(df_only_na_scaled_without_impute_columns)
        df_only_na_scaled[self.columns_to_impute] = df_imputations
        df_only_na = pd.DataFrame(self.scaler.inverse_transform(df_only_na_scaled),
                                      columns=df_only_na.columns,
                                      index=df_only_na.index)
        df_imputed = pd.concat([df_only_na, df_without_na], axis=0).sort_index()
        return df_imputed

    def save_model(self, filename: Union[str, WindowsPath]) -> None:
        with open(str(filename) + "_scaler", "wb") as file:
            pickle.dump(self.scaler, file)

        self.model.save(filename)
        print(f"Imputer model saved locally in: {filename}")

    def load_model(self, filename: Union[str, WindowsPath]) -> None:
        with open(str(filename) + "_scaler", "rb") as file:
            self.scaler = pickle.load(file)
        self.model = tf.keras.models.load_model(filename, compile=False)
