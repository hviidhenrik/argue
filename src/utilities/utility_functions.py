from typing import Union
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame

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


def plot_column_as_timeseries(df_column: DataFrame, title: str = None, save_path: Union[str, WindowsPath] = None
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
    fig = df_column.plot(rot=20)
    plt.ylabel(name)
    plt.xlabel("Time")
    if title is None:
        title = f"Time series of {name} over time"
    plt.title(title)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    return fig