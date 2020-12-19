from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import keras
import pickle
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense
from keras.models import Model, Sequential
from numpy.random import normal
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, average_precision_score, roc_auc_score, roc_curve
from pdm.models.clustering.dbscan import DBSCANClustering
from dataclasses import dataclass

@dataclass
class DataProvider:

    _scaler = None
    x_train = None
    x_val = None
    x_test = None

    def get_local_pump_data(self, filename="train-data-large.csv", dropna=True):
        df = pd.read_csv(filename, index_col="timelocal")
        if dropna:
            df = df.dropna()
        return df

    def train_val_test_split(self, df, train_size=0.7, val_size=0.2, test_size=0.1, shuffle=False):
        x_train, x_test = train_test_split(df, test_size=val_size + test_size, shuffle=shuffle)
        x_test, x_val = train_test_split(x_test, test_size=val_size / (val_size + test_size), shuffle=shuffle)
        self.x_train = x_train
        self.x_val = x_val
        self.x_test = x_test
        return x_train, x_val, x_test

    def scale(self, df=None, scaler=MinMaxScaler()):
        if self._scaler is None:
            self._scaler = scaler
        if df is None:
            x_train_scaled = pd.DataFrame(self._scaler.fit_transform(self.x_train), columns=self.x_train.columns)
            x_val_scaled = pd.DataFrame(self._scaler.transform(self.x_val), columns=self.x_train.columns)
            x_test_scaled = pd.DataFrame(self._scaler.transform(self.x_test), columns=self.x_train.columns)
            return x_train_scaled, x_val_scaled, x_test_scaled
        else:
            df_scaled = pd.DataFrame(self._scaler.fit_transform(df), columns=df.columns)
            return df_scaled

    def get_scaler(self):
        return self._scaler