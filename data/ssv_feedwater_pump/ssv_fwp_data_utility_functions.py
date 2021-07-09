import logging
import pickle
from typing import Union, Any, Tuple

import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from src.config.definitions import *
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from scipy import stats
from sklearn.model_selection import train_test_split
from src.config.definitions import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def get_data_size_from_execution_mode(debugging: bool = False) -> bool:
    """
    Simply returns a boolean indicator indicating whether running in debugging/testing mode or live as in
    training models to register and deploy. This is meant to tell functions which dataset to fetch for each
    purpose. See e.g. get_cleaned_pump_data in src/data/SSV/utils.py, which needs to know the size of
    the dataset.

    :param debugging: debugging/test mode or not
    :return: True if debugging/testing, False if not
    """
    return debugging


def get_dataset_size(small_dataset: Union[bool, str]) -> Union[str, bool]:
    """
    Gets the string size "large" or "small" of the dataset based on a provided boolean value indicating
    if a small dataset is used. This same string can also be used as argument to get the corresponding boolean.

    :param small_dataset: boolean/str value indicating if the dataset is small or not.
    :return: string/bool value corresponding to dataset size, i.e. either "small"/"large" or True/False, respectively
    """
    if isinstance(small_dataset, bool):
        return "small" if small_dataset else "large"
    else:
        return True if small_dataset == "small" else False


def get_other_pump_number(pump_number: Union[str, int]) -> str:
    """
    Simply returns the number of the other pump. I.e. if the current pump being modelled is 20, return 30.

    :param pump_number: the pump currently being modelled as a string - either "20" or "30"
    :return: the other pump number as a string.
    """
    pump_number = str(pump_number)
    if pump_number == "20":
        return "30"
    elif pump_number == "30":
        return "20"
    else:
        raise ValueError("Bad input given. Pump number must be either 20 or 30!")


def get_clean_figures_path(pump_number: Union[str, int]) -> Path:
    """
    Get the path where the data cleaning figures should be saved

    :param pump_number: the pump number of the pump being modelled (either 20 or 30)
    :return: a pathlib.Path object with the path to the folder
    """
    pump_number = str(pump_number)
    assert pump_number in ["20", "30"]
    return get_cleaning_figures_path() / f"ssv_fwp_{pump_number}"


def plot_effect_vs_flow(
    df, pump_number: Union[int, str], title_addendum: str = "", save: bool = False
) -> None:
    """
    Plots the effect of the given pump vs the flow in the intermediate area (area between the booster
    and the high pressure pump).

    :param df: dataframe with relevant data for the pump
    :param pump_number: pump number of the pump being modelled
    :param title_addendum: optional string to add to the title of the plot. Also added to filename if saved
    :param save: save the plot or not - will be saved in the reports/ folder
    """
    pump_number = str(pump_number)
    pump_other_number = "20" if pump_number == "30" else "30"
    fig = plt.scatter(
        df[f"effect_pump_{pump_number}_MW"],
        df["flow_after_pump"],
        s=8,
        c=df[f"effect_pump_{pump_other_number}_MW"],
        cmap="rainbow",
    )
    # plt.tight_layout()
    plt.suptitle(
        f"Pump {pump_number} effect vs flow {title_addendum}\n{df.index.min()} to {df.index.max()}"
    )
    clb = plt.colorbar()
    clb.set_label(
        f"effect_pump_{pump_other_number}_MW", rotation=0, labelpad=-30, y=1.05
    )
    plt.xlabel(f"Pump {pump_number} power [MW]")
    plt.ylabel(f"Mass flow after pump (pump {pump_number}) [kg/s]")
    plt.margins(0.2)
    plt.subplots_adjust(bottom=0.15)
    if save:
        title_addendum = title_addendum.replace(" ", "_")
        title_addendum = title_addendum.replace("\n", "_")
        save_path = get_clean_figures_path(pump_number)
        plt.savefig(
            save_path
            / f"scatter_pump_{pump_number}_effect_vs_flow_{title_addendum}.png"
        )
        plt.close()
    else:
        return fig


def plot_column_as_timeseries(
    df_column,
    column_name: str,
    pump_number: Union[str, int],
    title_addendum: str = "",
    save: bool = False,
) -> None:
    """
    Plot a given dataframe column as a timeseries.

    :param df_column: the dataframe column to plot
    :param column_name: the name of the column as a string
    :param pump_number: number of the pump being modelled
    :param title_addendum: optional string to add to the title of the plot. Also added to filename if saved
    :param save: should plot be saved or not
    """
    pump_number = str(pump_number)

    fig = df_column.plot(rot=20)
    plt.title(f"Pump {pump_number} tag {column_name} - {title_addendum}")
    plt.tight_layout()
    if save:
        title_addendum = title_addendum.replace(" ", "_")
        title_addendum = title_addendum.replace("\n", "_")
        save_path = get_clean_figures_path(pump_number)
        plt.savefig(
            save_path
            / f"timeseries_pump_{pump_number}_tag_{column_name}_{title_addendum}.png"
        )
        plt.close()
    else:
        return fig


def estimate_normal_operation_kde_from_effect_vs_flow(
    df,
    pump_number: Union[int, str],
    subsample_size: Union[float, int] = None,
    verbose: bool = True,
) -> Any:
    """
    Estimate a probability density for the relationship between effect and intermediate flow using a
    kernel density estimator.

    :param df: dataframe with the desired data - the needed columns will be picked automatically
    :param pump_number: pump number of the pump being modelled
    :param subsample_size: size of the subsample to use instead of the potentially LARGE original dataset.
            Can be either a fraction (0,1) or an absolute number e.g. 5000. Defaults to 1, i.e. full dataset.
    :param verbose: should the estimation be informative or not
    :return: the fitted KDE object
    """
    pump_number = str(pump_number)
    df_copy = df.copy()
    if subsample_size is not None and subsample_size != 1:
        df_copy, _ = train_test_split(df_copy, train_size=subsample_size)

    points = np.vstack(
        [df_copy[f"effect_pump_{pump_number}_MW"], df_copy["flow_intermediate_area"]]
    )
    if verbose:
        print("Estimating KDE density function...")

    return stats.gaussian_kde(points)


def clean_data_using_kde(
    df, kernel, pump_number, quantile: float = 0.02, verbose: bool = True
) -> Tuple[DataFrame, DataFrame]:
    """
    Takes a fitted KDE and cleans the data by removing datapoints with probability density
    lower than the threshold given in the quantile argument.

    :param df: the dataframe with the data to be cleaned - usually the same used for fitting the KDE.
    :param kernel: the fitted KDE object
    :param pump_number: number of the pump being modelled
    :param quantile: the density threshold at which to filter out datapoints with density lower than this
    :param verbose: should the cleaning be informative or not
    :return: a dataframe with the cleaned data and another with the datapoints deemed noise
    """
    df_copy = df.copy()
    N_rows_original = df_copy.shape[0]
    df_copy["pdf"] = kernel.evaluate(
        df_copy[[f"effect_pump_{pump_number}_MW", "flow_intermediate_area"]].T
    )

    if verbose:
        print("Separating normal points from noise...")
    quantile_cutoff_value = np.quantile(df_copy["pdf"], quantile)
    df_normal = df_copy[df_copy["pdf"] >= quantile_cutoff_value].drop("pdf", axis=1)
    df_noise = df_copy[df_copy["pdf"] < quantile_cutoff_value].drop("pdf", axis=1)
    N_rows_noise = df_noise.shape[0]
    if verbose:
        print(
            f"Rows removed: {N_rows_noise} of the initial {N_rows_original} "
            f"({100 * N_rows_noise / N_rows_original:2.2f} %)"
        )
    return df_normal, df_noise


def plot_kde_effect_vs_intermediate_flow(
    df_normal,
    df_noise,
    pump_number: Union[int, str],
    title_addendum: str = "",
    save: bool = False,
) -> None:
    """
    Plots the result of the KDE cleaning. Two dataframes are given, one with the cleaned data and one
    only with the datapoints considered noise by the KDE.

    :param df_normal: the dataframe with cleaned data
    :param df_noise: the dataframe with noisy data outside the KDE threshold
    :param pump_number: number of the pump being modelled
    :param title_addendum: optional string to add to the title of the plot. Also added to filename if saved
    :param save: should the plot be saved or not
    """
    pump_number = str(pump_number)
    plt.scatter(
        df_normal[f"effect_pump_{pump_number}_MW"],
        df_normal["flow_intermediate_area"],
        s=8,
        c="blue",
        label="Normal",
    )
    plt.scatter(
        df_noise[f"effect_pump_{pump_number}_MW"],
        df_noise["flow_intermediate_area"],
        s=8,
        c="red",
        label="Noise",
    )
    plt.tight_layout()
    plt.legend()
    plt.suptitle(
        f"Pump {pump_number} effect vs flow\nNormal vs noise points by "
        f"KDE estimation\n{title_addendum}"
    )
    plt.xlabel(f"Pump {pump_number} power [MW]")
    plt.ylabel(f"Mass flow intermediate area (pump {pump_number}) [kg/s]")
    if save:
        title_addendum = title_addendum.replace(" ", "_")
        title_addendum = title_addendum.replace("\n", "_")
        save_path = get_clean_figures_path(pump_number)
        plt.savefig(
            save_path
            / f"kde_noisepoints_pump_{pump_number}_effect_vs_intermediate_flow_KDE_{title_addendum}.png"
        )
        plt.close()
    else:
        plt.show()


def plot_slipring_temperature_difference(
    df: DataFrame,
    pump_number: Union[int, str],
    title_addendum: str = "",
    stddev_factor: float = 3,
    upper_quantile: float = 0.995,
    save: bool = False,
) -> None:
    """
    Plots the engineered feature "temperature difference" (temp_slipring_diff) as timeseries and some
    useful thresholds based on standard deviations and quantiles. Useful for spotting leakages.

    :param df: dataframe with data to plot - the needed column is automatically detected
    :param pump_number: number of the pump being modelled
    :param title_addendum: optional string to add to the title of the plot. Also added to filename if saved
    :param stddev_factor: how many standard deviations from the mean to put a dashed horizontal line
    :param upper_quantile: the data quantile at which to put another horizontal line
    :param save: save the plot or not
    """
    pump_number = str(pump_number)
    mean = np.mean(df["temp_slipring_diff"])
    stddev = np.std(df["temp_slipring_diff"])
    quantile_threshold = np.quantile(df["temp_slipring_diff"], upper_quantile)
    fig = df["temp_slipring_diff"].plot(rot=10)
    plt.hlines(
        mean, xmin=0, xmax=df.shape[0], colors="red", linestyles="solid", label=f"Avg"
    )
    plt.hlines(
        mean + stddev_factor * stddev,
        xmin=0,
        xmax=df.shape[0],
        colors="red",
        linestyles="dashed",
        label=f"{stddev_factor} x STDDEV",
    )
    plt.hlines(
        quantile_threshold,
        xmin=0,
        xmax=df.shape[0],
        colors="black",
        linestyles="dashed",
        label=f"{upper_quantile} quantile",
    )
    plt.legend()
    # plt.tight_layout()
    plt.title(
        f"Pump {pump_number} sliprings temperature difference\n"
        f"May indicate a leakage if very large\n"
        f"{title_addendum}"
    )
    plt.ylabel("Absolute temperature difference [Celsius]")
    if save:
        title_addendum = title_addendum.replace(" ", "_")
        title_addendum = title_addendum.replace("\n", "_")
        save_path = get_clean_figures_path(pump_number)
        plt.savefig(
            save_path
            / f"slipring_pump_{pump_number}_slipring_temp_diff_{title_addendum}.png"
        )
        plt.close()
    else:
        return fig


def plot_correlation_matrix(
    df, pump_number: Union[int, str], title_addendum: str = "", save: bool = False
) -> None:
    """
    Plots the correlation matrix of columns in a given dataframe.

    :param df: the dataframe with columns whose correlations should be plotted
    :param pump_number: number of the pump being modelled
    :param title_addendum: optional string to add to the title of the plot. Also added to filename if saved
    :param save: save the plot or not
    """
    pump_number = str(pump_number)
    corr = df.corr()
    plt.imshow(corr, cmap="seismic", interpolation="nearest")
    plt.colorbar()
    tick_marks = [i for i in range(len(df.columns))]
    plt.xticks(tick_marks, df.columns, rotation="vertical")
    plt.yticks(tick_marks, df.columns)
    plt.title("Correlation matrix")
    if save:
        title_addendum = title_addendum.replace(" ", "_")
        title_addendum = title_addendum.replace("\n", "_")
        save_path = get_clean_figures_path(pump_number)
        plt.savefig(save_path / f"correlations_pump_{pump_number}_{title_addendum}.png")
        plt.close()
    else:
        plt.show()


def _determine_active_pumps_as_df(
    df: DataFrame, mw_activity_threshold: float = 1
) -> DataFrame:
    """
    Determines when which pumps are active and returns a dataframe with the result.

    :param df: the dataframe containing effects for the three pumps 10, 20 and 30
    :param mw_activity_threshold: the megawatt threshold at which to deem a pump active
    :return: dataframe with binary columns indicating activity for each pump
    """
    df_copy = df.copy()
    pump_10_effect_column = "effect_pump_10_MW"
    pump_20_effect_column = "effect_pump_20_MW"
    pump_30_effect_column = "effect_pump_30_MW"
    df_copy["10_active"] = (
        df_copy[pump_10_effect_column] > mw_activity_threshold
    ).astype(int)
    df_copy["20_active"] = (
        df_copy[pump_20_effect_column] > mw_activity_threshold
    ).astype(int)
    df_copy["30_active"] = (
        df_copy[pump_30_effect_column] > mw_activity_threshold
    ).astype(int)
    df_copy["number_of_pumps_active"] = (
        df_copy["10_active"] + df_copy["20_active"] + df_copy["30_active"]
    )
    return df_copy[["10_active", "20_active", "30_active", "number_of_pumps_active"]]


def get_pump_dataframes_parallel_vs_single_operation(
    df: DataFrame, mw_activity_threshold: float = 0.9
) -> Tuple[DataFrame, DataFrame, DataFrame]:
    """
    Takes a dataframe of pump data containing, importantly, the effect used by the three
    pumps 10, 20 and 30 and returns a dataframe where only pump 20 was active and same for 30 as well
    as a dataframe where both were active at the same time.

    :param df: dataframe with the data for the three pumps (the effect of pump 10 must be present)
    :param mw_activity_threshold: threshold at which to deem a pump active
    :return: dataframes with data relating to only one pump active and both active
    """
    df_copy = df.copy()
    df_activity = _determine_active_pumps_as_df(
        df_copy, mw_activity_threshold=mw_activity_threshold
    )
    df_copy = pd.concat([df_copy, df_activity], axis=1)

    df_pump_20_active_only = df_copy[
        (df_copy["20_active"] == 1)
        & (df_copy["10_active"] == 0)
        & (df_copy["30_active"] == 0)
    ].drop(["10_active", "20_active", "30_active", "number_of_pumps_active"], axis=1)
    df_pump_30_active_only = df_copy[
        (df_copy["20_active"] == 0)
        & (df_copy["10_active"] == 0)
        & (df_copy["30_active"] == 1)
    ].drop(["10_active", "20_active", "30_active", "number_of_pumps_active"], axis=1)
    df_both_active = df_copy[
        (df_copy["10_active"] == 0)
        & (df_copy["20_active"] == 1)
        & (df_copy["30_active"] == 1)
    ].drop(["10_active", "20_active", "30_active", "number_of_pumps_active"], axis=1)
    return df_pump_20_active_only, df_pump_30_active_only, df_both_active


# def save_kde(kernel, filename: str) -> None:
#     """
#     Simply saves an esimated KDE object locally for later use.
#
#     :param kernel: the KDE to save
#     :param filename: filename under which to save it
#     """
#     filename = PUMP_DENSITY_ESTIMATORS_PATH / filename
#     with open(filename, "wb") as file:
#         pickle.dump(kernel, file)
#     print(f"KDE saved locally in {filename}")
#
#
# def load_kde(filename: str) -> Any:
#     """
#     Loads a saved and fitted KDE object for cleaning data.
#
#     :param filename: filename under which to find the saved KDE
#     :return: the loaded KDE object
#     """
#     filename = PUMP_DENSITY_ESTIMATORS_PATH / filename
#     with open(filename, "rb") as file:
#         kernel = pickle.load(file)
#     print(f"KDE loaded locally from {filename}")
#     return kernel


def filter_data_on_tail_quantiles(
    df_to_filter: DataFrame,
    upper_quantile: float = 0.995,
    lower_quantile: float = 0.005,
) -> DataFrame:
    """
    TODO consider making this work on standard deviations instead as it won't remove datapoints
      no matter what as the quantiles will as there will always be data beyond a given quantile.
    Takes a dataframe and removes data outside a given quantile for each column. Removes the
    most distant of noise points / outliers.

    :param df_to_filter: the dataframe whose columns should be filtered.
    :param upper_quantile: upper quantile to remove data beyond
    :param lower_quantile: lower quantile to remove data beyond
    :return: the cleaned dataframe
    """
    df_filtered = df_to_filter.copy()
    for column in df_filtered:
        df_filtered = df_filtered[
            (df_filtered[column] < np.quantile(df_filtered[column], upper_quantile))
            & (df_filtered[column] > np.quantile(df_filtered[column], lower_quantile))
        ]
    return df_filtered


def filter_points_beyond_linear_cut(df, x_col: str, y_col:str, slope: float,
                                    intercept: float = 0, remove_below_cut: bool = True):
    df_copy = df.copy()
    if remove_below_cut:
        mask = (intercept + slope * df_copy[x_col] > df_copy[y_col])
    else:
        mask = (intercept + slope * df_copy[x_col] < df_copy[y_col])
    return df_copy.drop(df_copy[mask].index)


def apply_SME_bounds_filter(
    df: DataFrame,
    pump_number: Union[int, str],
    megawatt_lower: float = 1.1,
    megawatt_upper: float = 10,
    flow_lower_bound: float = 18,
    flow_upper_bound: float = 200,
) -> DataFrame:
    """
    Applies apriori known boundaries on the data, that the pump should realistically be running within
    during normal operation based on SME recommendations.

    :param df: The dataframe with the raw data to be filtered
    :param pump_number: number of the pump at the station, usually 20 or 30
    :param megawatt_lower: lower pump effect boundary in megawatts
    :param megawatt_upper: upper pump effect boundary in megawatts
    :param flow_lower_bound: the minimum flow [kg/s] that should always be produced by
    the pump if running normally
    :param flow_upper_bound: the maximum flow [kg/s] that can be produced by
    the pump if running normally
    :return: filtered dataframe
    """
    df_copy = df.copy()
    df_copy = df_copy[
        (df_copy[f"effect_pump_{pump_number}_MW"] > megawatt_lower)
        & (df_copy[f"effect_pump_{pump_number}_MW"] < megawatt_upper)
    ]
    df_copy = df_copy[(df_copy["flow_after_pump"] > flow_lower_bound) &
                      (df_copy["flow_after_pump"] < flow_upper_bound)]
    N_before = df.shape[0]
    N_after = df_copy.shape[0]
    logger.info(
        f"Boundary filter: {N_before - N_after} rows out of {N_before} outside bounds and filtered away."
    )
    return df_copy


def make_feature_engineered_tags(df: DataFrame) -> DataFrame:
    """
    Simply takes a dataframe with the pump data and makes some engineered tags based on temperature
    of the water between the slipring, as well as averages the two "flow after pump" measurings.
    The two flow after pump measurings are dropped and only the average retained as "flow_after_pump"

    :param df: dataframe with the data to have feature engineered tags added
    :return: the engineered dataframe
    """
    df_copy = df.copy()
    col_names = df_copy.columns.values
    if "temp_slipring_diff" not in col_names and "flow_after_pump" not in col_names:
        logger.info("Doing feature engineering: calculating slipring water temperature difference\n"
                    "and averaging flow measurings after the pump")
        df_copy["temp_slipring_diff"] =\
            abs(df_copy["temp_slipring_water_pressure_side"] - df_copy["temp_slipring_water_suction_side"])
        if "flow_after_pump_1" in col_names and "flow_after_pump_2" in col_names:
            df_copy["flow_after_pump"] = (df_copy["flow_after_pump_1"] + df_copy["flow_after_pump_2"]) / 2
            df_copy = df_copy.drop(columns=["flow_after_pump_1", "flow_after_pump_2"])
        elif "flow_after_pump_1" in col_names:
            df_copy["flow_after_pump"] = df_copy["flow_after_pump_1"]
        elif "flow_after_pump_2" in col_names:
            df_copy["flow_after_pump"] = df_copy["flow_after_pump_2"]
        else:
            raise Exception("No flow after pump detected in data, tag is missing!")
    else:
        logger.info(
            "Feature engineered tags already present, no feature engineering done!"
        )
    return df_copy


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


# def local_data_exists(
#     small_dataset: bool, pump_number: Union[str, int], already_cleaned: bool
# ) -> bool:
#     """
#     Helper function to determine if already cleaned data for training exists locally. It checks the folder
#     src/data/SSV for the presence of a .csv file containing already cleaned data.
#
#     :param small_dataset: boolean indicator of whether the dataset is small or large
#     :param pump_number: the number of the pump being modelled
#     :param already_cleaned: boolean indicating if the data to look for should be raw data or already cleaned
#     :return: boolean indicating if local cleaned data exists in the src/data/SSV folder or not
#     """
#     cleaned_or_raw_addendum = "_cleaned" if already_cleaned else ""
#     return Path(
#         get_data_path() / "SSV" / f"data_pump_{pump_number}_"
#         f"{get_dataset_size(small_dataset)}{cleaned_or_raw_addendum}.csv"
#     ).is_file()


def get_local_pump_data(
    small_dataset=True, pump_number: str = None, filename: str = None
) -> DataFrame:
    """
    Fetches local pump data. By default will look for a filename by the following string:
    "data_pump_{pump_number}_{dataset_size}.csv" --> "data_pump_20_large.csv", for example.

    :param small_dataset: boolean indicator whether the datasize is small or not
    :param pump_number: number of the pump being modelled
    :param filename: optional filename if data is saved in an unrecognized filename
    :return: the loaded dataframe
    """
    dataset_size = "small" if small_dataset else "large"
    filename = (
        f"data_pump_{pump_number}_{dataset_size}.csv" if filename is None else filename
    )
    filename = filename if filename.endswith(".csv") else filename.join(".csv")
    filename = get_data_path() / "ssv_feedwater_pump" / filename
    df = pd.read_csv(filename, index_col="timelocal")
    print(f'Local data loaded from file: "{filename}"')
    return df


def get_cleaned_pump_data(
    small_dataset=True, pump_number: str = None, filename: str = None
) -> DataFrame:
    """
    Fetches local pump data that has been cleaned. By default will look for a filename by
    the following string:
    "data_pump_{pump_number}_{dataset_size}_cleaned.csv" --> "data_pump_20_large_cleaned.csv", for example.

    :param small_dataset: boolean indicator whether the datasize is small or not
    :param pump_number: number of the pump being modelled
    :param filename: optional filename if data is saved in an unrecognized filename
    :return: the loaded dataframe
    """
    dataset_size = "small" if small_dataset else "large"
    filename = (
        f"data_pump_{pump_number}_{dataset_size}_cleaned.csv"
        if filename is None
        else filename
    )
    filename = filename if filename.endswith(".csv") else filename.join(".csv")
    filename = get_data_path() / "ssv_feedwater_pump" / filename
    df = pd.read_csv(filename, index_col="timelocal")
    print(f'Cleaned training data loaded from file: "{filename}"')
    return df


def save_local_pump_data(
    df_to_save,
    small_dataset=None,
    pump_number: str = None,
    default_filename_addendum: str = "",
    custom_filename: str = None,
) -> None:
    """
    Takes a dataframe and saves it to a local file. By default will save to a file with name:
    "data_pump_{pump_number}_{dataset_size}{default_filename_addendum}.csv"

    :param df_to_save: dataframe to save
    :param small_dataset: boolean indicating whether the dataset is small or not
    :param pump_number: number of the pump being modelled
    :param default_filename_addendum: optional extra string to add to the filename
    :param custom_filename: a completely custom filename entirely.
    """
    pump_number = str(pump_number)
    dataset_size = "small" if small_dataset else "large"
    if custom_filename is None:
        filename = (
            f"data_pump_{pump_number}_{dataset_size}{default_filename_addendum}.csv"
        )
    else:
        filename = custom_filename
    filename = filename if filename.endswith(".csv") else filename.join(".csv")
    filename = get_data_path() / "ssv_feedwater_pump" / filename
    df_to_save.to_csv(filename)
    print(f"Data saved locally as {filename}")


def make_and_save_small_dataset(
    df_large, pump_number: str = None, size: int = 800
) -> None:
    """
    Takes a large dataset and autoamtically creates a smaller one, useful for testing and debugging.
    Can also be used for parameter tuning, as the data is selected so it represents the trends in the
    large original dataset, so the most important aspects of the data will be caught if big enough.
    Will save the dataset to a local file.

    :param df_large: the original data to create a subset from
    :param pump_number: number of the pump being modelled
    :param size: size of the subset. Can either be a fraction or an absolute number. Defaults to 800.
    """
    pump_number = str(pump_number)
    df_small, _ = train_test_split(df_large, train_size=size)
    df_small = df_small.sort_index()
    save_local_pump_data(df_small, small_dataset=True, pump_number=pump_number)


def merge_dfs_and_drop_duplicate_indices(df1: DataFrame, df2: DataFrame) -> DataFrame:
    """
    Takes two dataframes and only merges the non-overlapping rows together in a new dataframe.

    :param df1: First dataframe to merge
    :param df2: Second dataframe to merge
    :return: new dataframe with the merged data
    """
    df_merged = pd.concat([df1, df2], axis=0)
    overlapping_rows_indicator = df_merged.index.duplicated()
    N_overlapping_rows = overlapping_rows_indicator.sum()
    percentage_overlap = 100 * N_overlapping_rows / df2.shape[0]
    df_merged = df_merged[~overlapping_rows_indicator].sort_index()
    print(
        f"Dataframes merged. Overlapping rows removed: {N_overlapping_rows} ({percentage_overlap:0.0f} %)"
    )
    return df_merged


# def remove_merged_in_data(file_to_move) -> None:
#     """
#     Moves new data that has already been merged with old data to another folder, so no confusion
#     as to whether the data has already been merged in should take place.
#
#     :param file_to_move: path to the file with the merged in data that should be moved
#     """
#     filename = file_to_move.name
#     old_merged_data_file = (
#         DATA_PATH / "old_merged_data" / (filename[:-8] + "_MERGED_IN.csv")
#     )
#     if old_merged_data_file.is_file():
#         old_merged_data_file.unlink()  # means delete it
#     file_to_move.rename(
#         DATA_PATH / "old_merged_data" / (filename[:-8] + "_MERGED_IN.csv")
#     )
#
#
# def time_interval_overlaps_with_df(df: DataFrame, time_interval: time_interval) -> bool:
#     """
#     Checks if a given dataframe and time interval have overlaps in time
#
#     :param df: The dataframe to check against
#     :param time_interval: the time interval to check with
#     :return: boolean indicator, True if overlap, False if not
#     """
#     start_date_old = pd.to_datetime(df.index.min())
#     end_date_old = pd.to_datetime(df.index.max())
#     start_date_new = pd.to_datetime(time_interval[0])
#     end_date_new = pd.to_datetime(time_interval[1])
#     overlaps = (start_date_old < end_date_new) and (start_date_new < end_date_old)
#     return overlaps

