from typing import Union, List

import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from argue.config.definitions import *


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


def put_sample_and_faulty_cols_first(df):
    cols = df.columns.values.tolist()
    assert "sample" in cols, "\"sample\" column not in the dataframe"
    assert "faulty" in cols, "\"faulty\" column not in the dataframe"
    cols = cols[-2:] + cols[:-2]
    return df[cols]


def make_sample_and_faulty_cols(df):
    df_copy = df.copy()
    df_copy["sample"] = [i for i in range(1, df_copy.shape[0] + 1)]
    df_copy["faulty"] = [0 for _ in range(1, df_copy.shape[0] + 1)]
    df_copy = put_sample_and_faulty_cols_first(df_copy)
    return df_copy
