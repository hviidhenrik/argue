"""
Misc utilities, helper functions etc
"""

import random
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import tensorflow as tf
from pandas import DataFrame
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


def set_seed(seed: int = 1234):
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)


def vprint(verbose: Union[bool, int], str_to_print: str, **kwargs):
    if verbose:
        print(str_to_print, **kwargs)


def partition_by_quantiles(x, column: str, quantiles: List[float] = None):
    if quantiles is None:
        quantiles = [0, 0.25, 0.5, 0.75, 1.0]
    bins = pd.qcut(x[column], quantiles, labels=False)
    x_out = x.copy()
    x_out["partition"] = bins + 1  # add one since the partitions to ARGUE must start in 1
    return x_out


def reduce_dimension_by_pca(x):
    """
    This is a helper function that simply does a PCA and returns the pca object and transformed data. Use before
    plot_candidate_partitions_by_pca or select_pcs_and_partition_data
    """
    pca = PCA().fit(x)
    x_transformed = pca.transform(x)
    return x_transformed, pca


def plot_candidate_partitions_by_pca(x_transformed, pca):
    """
    This function is a visual aid to guide the number of partitions and which principal components to
    use for clustering. Should be followed by select_pcs_and_partition_data called on the output from this
    using the visual conclusions obtained from this one. Plots adapted from:
    https://plotly.com/python/pca-visualization/
    """

    # pca = PCA().fit(x)
    # x_transformed = pca.transform(x)
    components = x_transformed
    labels = {str(i): f"PC {i + 1} ({var:.1f}%)" for i, var in enumerate(pca.explained_variance_ratio_ * 100)}
    dimensions = range(np.min([5, x_transformed.shape[1]]))  # get available PC's up to a max of 5

    fig = px.scatter_matrix(components, labels=labels, dimensions=dimensions,)
    fig.update_traces(diagonal_visible=False)
    fig.show()
    return x_transformed


def select_pcs_and_partition_data(
    x_transformed, pcs_to_cluster_on: List[int], n_clusters: int, plot_pca_clustering: bool = True
):
    """
    This function relies on the visual conclusions from plot_candidate_partitions_by_pca.
    The particular PC's to cluster on can be selected as well as the desired number of clusters/partitions to obtain
    """

    assert 0 not in pcs_to_cluster_on, "PC's must be given by indices starting from 1"
    pcs_to_cluster_on = [i - 1 for i in pcs_to_cluster_on]  # convert to actual array indices
    clusters = KMeans(n_clusters=n_clusters, n_init=50).fit(x_transformed[:, pcs_to_cluster_on])
    partition_labels = clusters.labels_ + 1
    if plot_pca_clustering:
        components = x_transformed
        dimensions = range(np.min([5, x_transformed.shape[1]]))  # get available PC's up to a max of 5
        labels = {str(i): f"PC {i+1}" for i in dimensions}

        fig = px.scatter_matrix(components, labels=labels, dimensions=dimensions, color=partition_labels)
        fig.update_traces(diagonal_visible=False)
        fig.show()
    return partition_labels


# def partition_by_pca_and_clustering(x, n_pca_components: int = 2, n_clusters: int = 2, plot_pca_clustering: bool = False):
#     """
#     DEPRECATED. Use plot_candidate_partitions_by_pca followed by select_pcs_and_partition_data instead.
#     """
#     pca = PCA(n_pca_components).fit(x)
#     x_transformed = pca.transform(x)
#     explained_variance = np.round(pca.explained_variance_ratio_.cumsum(), 2)
#     print("Variance explained by PC's:")
#     print(explained_variance)
#     clusters = KMeans(n_clusters=n_clusters, n_init=10).fit(x_transformed[:,[1,2]])
#     x["partition"] = clusters.labels_ + 1
#     if plot_pca_clustering:
#         components = x_transformed
#         labels = {
#             str(i): f"PC {i + 1} ({var:.1f}%)" for i, var in enumerate(pca.explained_variance_ratio_ * 100)
#         }
#         dimensions = range(5) if n_pca_components >= 5 else range(n_pca_components)
#
#         fig = px.scatter_matrix(
#             components,
#             labels=labels,
#             dimensions=dimensions,
#             color=x["partition"]
#         )
#         fig.update_traces(diagonal_visible=False)
#         fig.show()
#     return x


def check_alarm_sparsity(y_true, y_pred):
    absolute_error = np.abs(np.sum(y_true, axis=1) - np.sum(y_pred, axis=1))
    return np.mean(absolute_error)


# note: authors' original noise generator
def generate_noise_samples(
    x: DataFrame, n_noise_samples: Optional[int] = None, mean: float = 0.5, stdev: float = 1,
):
    input_dim = pd.DataFrame(x).shape[1]
    # if no noise samples desired, make a df with a single row to keep dimensions intact further on in ARGUE
    if n_noise_samples == 0:
        return pd.DataFrame(np.random.normal(size=(1, input_dim)), columns=x.columns)
    N = x.shape[0] if n_noise_samples is None else n_noise_samples
    noise = np.random.normal(mean, stdev, size=(N, input_dim))
    df_noise = pd.DataFrame(noise, columns=x.columns)
    return df_noise


# note: my modified noise generator
def generate_noise_samples2(
    x: DataFrame,
    quantiles: Optional[List[float]] = None,
    n_noise_samples: Optional[int] = None,
    stdev: float = 1,
    stdevs_away: float = 3,
):
    input_dim = pd.DataFrame(x).shape[1]

    # if no noise samples desired, make a df with a single row to keep dimensions intact further on in ARGUE
    if n_noise_samples == 0:
        return pd.DataFrame(np.random.normal(size=(1, input_dim)), columns=x.columns)

    quantiles = [0.025, 0.975] if quantiles is None else quantiles
    N = x.shape[0] // 2 if n_noise_samples is None else n_noise_samples // 2
    qs = x.quantile(quantiles)
    noise_below = np.random.normal(qs.iloc[0] - stdev * stdevs_away, stdev, size=(N, input_dim))
    noise_above = np.random.normal(qs.iloc[1] + stdev * stdevs_away, stdev, size=(N, input_dim))
    df_noise = pd.DataFrame(np.vstack((noise_below, noise_above)), columns=x.columns)
    return df_noise


# note: another modified noise generator
def generate_noise_samples3(x: DataFrame, n_noise_samples: Optional[int] = None, **kwargs):
    input_dim = pd.DataFrame(x).shape[1]
    n_noise_samples = x.shape[0] // 2 if n_noise_samples is None else n_noise_samples // 2

    # if no noise samples desired, make a df with a single row to keep dimensions intact further on in ARGUE
    if n_noise_samples < 2:
        return pd.DataFrame(np.random.normal(size=(1, input_dim)), columns=x.columns)

    normal_area = (x.min().min(), x.max().max())

    df_noise1 = np.random.normal(size=(n_noise_samples, input_dim))
    df_noise2 = np.random.normal(size=(n_noise_samples, input_dim))
    noise_below = MinMaxScaler(feature_range=(-6, normal_area[0] - 1)).fit_transform(df_noise1)
    noise_above = MinMaxScaler(feature_range=(normal_area[1] + 1, 6)).fit_transform(df_noise2)
    df_noise = pd.DataFrame(np.vstack((noise_below, noise_above)), columns=x.columns)
    return df_noise


def plot_learning_schedule(
    total_steps: int = None,
    initial_learning_rate: float = None,
    decay_rate: float = None,
    decay_steps: int = None,
    staircase: bool = False,
    verbose: bool = False,
):
    def decayed_learning_rate(step):
        exponent = (step // decay_steps) if staircase else (step / decay_steps)
        return np.power(initial_learning_rate * decay_rate, exponent)

    def get_lr(step):
        return decayed_learning_rate(step) if step > decay_steps else initial_learning_rate

    steps = np.arange(total_steps)
    lr = []
    for step in steps:
        learning_rate = get_lr(step)
        lr.append(learning_rate)
        vprint(verbose, f"Learning rate: {learning_rate:.4f}")

    plt.plot(steps, lr)
    plt.suptitle(f"Learning rate schedule with decay_rate = {decay_rate}")
    plt.xlabel("Step")
    plt.ylabel("Learning rate")


def make_time_elapsed_string(elapsed_time, secs_to_min_threshold: int = 180):
    return (
        f"{elapsed_time:.2f} seconds!" if elapsed_time < secs_to_min_threshold else f"{elapsed_time / 60:.2f} minutes!"
    )
