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
from src.models.AnoGen.utility_functions import fit_VAE


class Visualizer:
    def plot_vae_latent(self, color_by_columns=None, save=False, show=True, ax=None):
        if color_by_columns is None:
            color_by_columns = [self.df_vae_train_data.columns[0]]
        elif type(color_by_columns) != list:
            if color_by_columns.lower() == "all":
                color_by_columns = self.df_vae_train_data.columns.values
            else:
                color_by_columns = [color_by_columns]  # if only a single string is received like "kv_flow"
        df = self.df_vae_latent_space.copy()
        pca_reduce = False
        title_latent = "AnoGen VAE training latent space"
        if self.vae_latent_dim > 2:
            pca = PCA(n_components=2).fit(df)
            df = pd.DataFrame(pca.fit_transform(df))
            var_expl = 100 * pca.explained_variance_ratio_.sum()
            title_latent = title_latent + "\nPCA reduced (var explained: {0:4.0f})%".format(var_expl)
            pca_reduce = True
            self.vae_latent_pca = pca

        for coloring_col in color_by_columns:
            plt.scatter(df.iloc[:, 0], df.iloc[:, 1],
                        c=self.df_vae_train_data[coloring_col], cmap='jet', s=10)
            plt.xlabel("PC1" if pca_reduce else "z0")
            plt.ylabel("PC2" if pca_reduce else "z1")
            clb = plt.colorbar()
            clb.set_label(coloring_col, rotation=0, labelpad=-30, y=1.05)
            plt.title(title_latent)
            if save:
                plt.savefig('VAE_latent_[{}].png'.format(coloring_col))
            if show:
                plt.show()
