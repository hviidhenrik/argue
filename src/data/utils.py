from typing import Union, List

import pandas as pd
import timesynth as ts
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from src.config.definitions import *


def get_local_data(filename: Union[str, Path] = None) -> DataFrame:
    """
    Fetches local pump data. By default will look for a filename by the following string:
    "data_pump_{pump_number}_{dataset_size}.csv" --> "data_pump_20_large.csv", for example.

    :param filename: filename for the file with the data, not needed if filename_contains is provided
    :return: the loaded dataframe
    """
    assert filename is not None, "No filename provided!"
    df = pd.read_csv(filename, index_col="timelocal")
    print(f'Local data loaded from file: "{filename}"')
    return df


def save_local_data(df_to_save: DataFrame, filename: Union[str, Path] = None, index=False) -> None:
    """
    Takes a dataframe and saves it to a local file. By default will save to a file with name:
    "data_pump_{pump_number}_{dataset_size}{default_filename_addendum}.csv"

    :param df_to_save: dataframe to save
    :param filename: optional filename for the file to save
    """
    filename = filename if str(filename).endswith(".csv") else filename.join(".csv")
    df_to_save.to_csv(filename, index=index)
    print(f"Data saved locally as {filename}")


def make_and_save_debugging_dataset(
        df_large, size: int = 800, filename: Union[str, Path] = None, index=None,
) -> None:
    """
    Takes a large dataset and automatically creates a smaller one, useful for testing and debugging.
    Can also be used for parameter tuning, as the data is selected so it represents the trends in the
    large original dataset, so the most important aspects of the data will be caught if big enough.
    Will save the dataset to a local file.

    :param df_large: the original data to create a subset from
    :param size: size of the subset. Can either be a fraction or an absolute number. Defaults to 800.
    :param filename: filename for the file to save

    """
    assert filename is not None, "No filename provided!"
    df_small, _ = train_test_split(df_large, train_size=size)
    df_small = df_small.sort_index()
    save_local_data(df_small, filename, index)


def get_dataset_purpose_as_str(debugging: bool = None) -> str:
    return "small" if debugging else "large"


def is_dataset_for_debugging(purpose: str = None) -> bool:
    return True if "debugging" in purpose else False


def get_df_with_bad_data(df_cleaned: DataFrame, df_not_cleaned: DataFrame) -> DataFrame:
    """
    Gets the set difference between df_cleaned and df_not_cleaned, where df_not_cleaned is supposed to be
    a bigger dataframe than df_cleaned. This gets the rows that were filtered out during cleaning.
    Useful for testing anomaly models on since most of these rows should be labeled as anomalies.

    :param df_cleaned: dataframe of cleaned data
    :param df_not_cleaned: dataframe of raw/uncleaned data
    :return: dataframe of rows that were filtered out during data cleaning
    """
    return df_not_cleaned[~df_not_cleaned.index.isin(df_cleaned.index)]


def make_univariate_timeseries(num_points: int = 100, noise_std: float = 1.0):
    time_sampler = ts.TimeSampler(stop_time=20)
    irregular_time_samples = time_sampler.sample_regular_time(num_points=num_points)
    ar = ts.signals.CAR()
    white_noise = ts.noise.GaussianNoise(std=noise_std)
    timeseries = ts.TimeSeries(ar, noise_generator=white_noise)
    samples, _, _ = timeseries.sample(irregular_time_samples)
    return pd.DataFrame(samples, columns=["x1"])


def make_class_labels(classes: int, N: int):
    labels = []
    for i in range(1, classes+1):
        labels += [i for _ in range(N)]
    return labels


def make_custom_test_data(N1, N2, N3, noise_sd: List[float] = [1, 10, 2]):
    df = pd.concat([
        pd.DataFrame({"x1": 4 + np.sin(np.linspace(0, 10, N1) + np.random.normal(0, noise_sd[0], N1)),
                      "x2": 4 + np.cos(np.linspace(0, 10, N1) + np.random.normal(0, noise_sd[0], N1)),
                      "x3": 4 + np.cos(3.14 + np.linspace(0, 10, N1) + np.random.normal(0, noise_sd[0], N1)),
                      }),
        pd.DataFrame({"x1": 500 + np.sin(np.linspace(0, 10, N2) + np.random.normal(0, noise_sd[1], N2)),
                      "x2": 500 + np.cos(np.linspace(0, 10, N2) + np.random.normal(0, noise_sd[1], N2)),
                      "x3": 500 + np.cos(3.14 + np.linspace(0, 10, N2) + np.random.normal(0, noise_sd[1], N2)),
                      }),
        pd.DataFrame({"x1": -100 - 2 * np.linspace(0, 10, N3) + np.random.normal(0, noise_sd[2], N3),
                      "x2": -100 - 3 * np.linspace(0, 10, N3) + np.random.normal(0, noise_sd[2], N3),
                      "x3": -100 - 1 * np.linspace(0, 10, N3) + np.random.normal(0, noise_sd[2], N3),
                      })
    ]).reset_index(drop=True)
    return df
