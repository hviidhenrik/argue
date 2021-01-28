from typing import Union, Tuple, Any, List
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame
import numpy as np

from src.config.definitions import *


def plot_missing_values_bar_chart(df: DataFrame, title: str = None, save_path: Union[str, WindowsPath] = None):
    """
    Plot the missing values in the dataset as a bar chart of percentage missing values.

    :param df: the dataframe with missing values
    :param title: optional title of the plot
    :param save_path: a string or WindowsPath form pathlib with the location to save the figure in. If not given
    the figure will not be saved
    :return: the figure object from matplotlib
    """
    df_NA = 100 * df.isna().sum(axis=0) / df.shape[0]
    fig = df_NA.plot.bar()
    if title is None:
        title = "Percentage NA's present in dataset"
    plt.suptitle(title)
    plt.ylabel("[% NA]")
    if save_path is not None:
        plt.savefig(save_path)
    return fig


def plot_missing_values_heatmap(df: DataFrame, title: str = None, save_path: Union[str, WindowsPath] = None):
    """
    Plot the missing values in the dataset as a heatmap of missing values.

    :param df: the dataframe with missing values
    :param title: optional title of the plot
    :param save_path: a string or WindowsPath form pathlib with the location to save the figure in. If not given
    the figure will not be saved
    :return: the figure object from matplotlib
    """
    fig = sns.heatmap(df.isnull(), cbar=False)
    if title is None:
        title = "Missing values heatmap"
    plt.suptitle(title)
    if save_path is not None:
        plt.savefig(save_path)
    return fig


def plot_column_as_timeseries(df_column: DataFrame,
                              title: str = None,
                              save_path: Union[str, WindowsPath] = None,
                              **kwargs
) -> None:
    """
    Plot a given dataframe column as a timeseries.

    :param df_column: the dataframe with missing values
    :param title: optional title of the plot
    :param save_path: a string or WindowsPath form pathlib with the location to save the figure in. If not given
    the figure will not be saved
    :return: the figure object from matplotlib
    """
    name = df_column.name
    fig = df_column.plot(rot=20, **kwargs)
    plt.ylabel(name)
    plt.xlabel("Time")
    if title is None:
        title = f"Time series of {name} over time"
    plt.title(title)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    return fig


def find_bin_with_most_values(df_column: DataFrame, interval_size: float) -> Tuple[float, float]:
    bins = np.arange(np.floor(df_column.min()), np.ceil(df_column.max()), interval_size)
    count, edges = np.histogram(np.array(df_column).reshape(-1, 1), bins=bins)
    edge_low = edges[np.argmax(count)]
    edge_high = edges[np.argmax(count)+1]
    return edge_low, edge_high



def find_all_nonempty_bins(df_column: DataFrame,
                           interval_size: float,
                           required_bin_size: int = 1
                           ) -> List[Tuple[float, float]]:
    bins = np.arange(np.floor(df_column.min()), np.ceil(df_column.max()), interval_size)
    count, edges = np.histogram(np.array(df_column).reshape(-1, 1), bins=bins)
    indices_nonempty = [i for i, count in enumerate(count) if count >= required_bin_size]
    non_empty_intervals = edges[indices_nonempty]
    edge_interval_list = []
    for index, interval_left in enumerate(non_empty_intervals):
        edge_interval_list.append((non_empty_intervals[index], non_empty_intervals[index]+interval_size))
    return edge_interval_list
