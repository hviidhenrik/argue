"""
This script cleans the data from feedwater pump 20 at SSV
"""
import logging
import sys
from src.models.SSV.utility_functions import *
from src.data.SSV.data_utils import *

PUMP_NUMBER = "20"
SMALL_DATASET = True
SAVE_PLOTS = True
USE_SAVED_KDE = False

if len(sys.argv) > 1:
    SMALL_DATASET = get_dataset_size(sys.argv[1])

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if __name__ == "__main__":
    plt.style.use("seaborn")
    PUMP_OTHER_NUMBER = get_other_pump_number(PUMP_NUMBER)
    DATASET_SIZE = get_dataset_size(SMALL_DATASET)

    # --------------------------
    logger.info("===== Feedwater pump 20 data cleaning script =====")

    # load data
    df = get_local_pump_data(small_dataset=SMALL_DATASET, pump_number=PUMP_NUMBER)
    df = df.dropna().drop_duplicates()

    logger.info(f"Raw data period: {df.index.min()} to {df.index.max()}")

    # choose initial data period
    df = df.loc["2019-08-13":]
    N_rows_original = df.shape[0]
    logger.info(f"Training data period: {df.index.min()} to {df.index.max()}")

    # plot raw data as it is
    plot_effect_vs_intermediate_flow(df, PUMP_NUMBER, "RAW DATA", save=SAVE_PLOTS)

    # set apriori known limits for normal operation cf. Rasmus and Jonas
    df = apply_SME_bounds_filter(df, PUMP_NUMBER, 1.1, 10, 18)
    df = make_feature_engineered_tags(df)
    df_not_cleaned = df.copy()

    # filter out some noisy outlier weirdness in the data
    df = df.drop(
        df[
            (df[f"effect_pump_{PUMP_NUMBER}_MW"] > 2)
            & (df[f"flow_intermediate_area"] < 45)
        ].index
    )

    N_rows_after_cleaning = df.shape[0]
    plot_effect_vs_intermediate_flow(
        df, PUMP_NUMBER, "AFTER BOUNDS FILTER", save=SAVE_PLOTS
    )

    plot_slipring_temperature_difference(df, pump_number=PUMP_NUMBER, save=SAVE_PLOTS)

    # inspection of the slipring temperature diff indicates that we should only use data
    # for temperature differences less than around 10 degrees to not get abnormal spikes
    df = df[df["temp_slipring_diff"] < 10]

    # split data into one pump vs two pumps running
    (
        df_pump_20_active,
        df_pump_30_active,
        df_both_active,
    ) = get_pump_dataframes_parallel_vs_single_operation(df, mw_activity_threshold=1.1)
    df_this_pump_active = (
        df_pump_20_active if PUMP_NUMBER == "20" else df_pump_30_active
    )
    df_other_pump_active = (
        df_pump_30_active if PUMP_NUMBER == "20" else df_pump_20_active
    )

    if df_this_pump_active.shape[0] < 1 or df_both_active.shape[0] < 1:
        raise Exception(
            "Error: no data for either mode one-pump-active or both-active! Try using some more data. "
            "Aborting script."
        )

    plot_effect_vs_intermediate_flow(
        df_this_pump_active, PUMP_NUMBER, f"ONLY {PUMP_NUMBER} ACTIVE", save=SAVE_PLOTS
    )
    plot_effect_vs_intermediate_flow(
        df_both_active, PUMP_NUMBER, "BOTH ACTIVE", save=SAVE_PLOTS
    )

    kde_filename_this_pump_active = (
        f"KDE_pump_{PUMP_NUMBER}_active_{DATASET_SIZE}_dataset.pkl"
    )
    kde_filename_both_active = f"KDE_pump_both_active_{DATASET_SIZE}_dataset.pkl"
    if USE_SAVED_KDE:
        kernel_this_pump_active = load_kde(kde_filename_this_pump_active)
        kernel_both_active = load_kde(kde_filename_both_active)
    else:
        kernel_this_pump_active = estimate_normal_operation_kde_from_effect_vs_flow(
            df_this_pump_active, PUMP_NUMBER, subsample_size=1
        )
        kernel_both_active = estimate_normal_operation_kde_from_effect_vs_flow(
            df_both_active, PUMP_NUMBER, subsample_size=1
        )
        save_kde(kernel_this_pump_active, filename=kde_filename_this_pump_active)
        save_kde(kernel_both_active, filename=kde_filename_both_active)

    df_normal_this, df_noise_this = clean_data_using_kde(
        df_this_pump_active, kernel_this_pump_active, PUMP_NUMBER, quantile=0.001
    )
    plot_kde_effect_vs_intermediate_flow(
        df_normal_this,
        df_noise_this,
        PUMP_NUMBER,
        title_addendum=f"ONLY {PUMP_NUMBER} ACTIVE",
        save=SAVE_PLOTS,
    )

    df_normal_both, df_noise_both = clean_data_using_kde(
        df_both_active, kernel_both_active, PUMP_NUMBER, quantile=0.00075
    )
    plot_kde_effect_vs_intermediate_flow(
        df_normal_both,
        df_noise_both,
        PUMP_NUMBER,
        title_addendum="BOTH ACTIVE",
        save=SAVE_PLOTS,
    )

    plot_effect_vs_intermediate_flow(
        df_normal_this,
        PUMP_NUMBER,
        f"AFTER KDE CLEANING\nONLY PUMP {PUMP_NUMBER} ACTIVE",
        save=SAVE_PLOTS,
    )
    plot_effect_vs_intermediate_flow(
        df_normal_both, PUMP_NUMBER, "AFTER KDE CLEANING\nBOTH ACTIVE", save=SAVE_PLOTS
    )

    # inspect temperature difference in water between sliprings (may indicate leak if very large)
    plot_slipring_temperature_difference(
        df_normal_this,
        pump_number=PUMP_NUMBER,
        title_addendum=f"ONLY {PUMP_NUMBER} ACTIVE",
        save=SAVE_PLOTS,
    )
    plot_slipring_temperature_difference(
        df_normal_both,
        pump_number=PUMP_NUMBER,
        title_addendum="BOTH ACTIVE",
        save=SAVE_PLOTS,
    )

    # do a final cleaning of the biggest outliers in each column
    logger.info("Filtering data on each column...")
    df_normal_this = filter_data_on_tail_quantiles(
        df_normal_this, upper_quantile=0.995, lower_quantile=0.005
    )
    df_normal_both = filter_data_on_tail_quantiles(
        df_normal_both, upper_quantile=0.995, lower_quantile=0.005
    )

    df_final = pd.concat([df_normal_this, df_normal_both], axis=0).sort_index()
    df_discarded_rows = get_df_with_bad_data(df_final, df_not_cleaned)

    # generate timeseries plots of all the columns
    logger.info("Plotting each cleaned column as separate timeseries...")
    for column in df_final:
        plot_column_as_timeseries(
            df_final[column],
            column,
            PUMP_NUMBER,
            "FINAL DF SINGLE AND BOTH",
            save=SAVE_PLOTS,
        )
    for column in [
        f"effect_pump_{PUMP_NUMBER}_MW",
        "flow_intermediate_area",
        "temp_slipring_diff",
    ]:
        plot_column_as_timeseries(
            df_discarded_rows[column],
            column,
            PUMP_NUMBER,
            "DISCARDED DATA",
            save=SAVE_PLOTS,
        )

    N_rows_after_cleaning = df_final.shape[0]
    logger.info(
        f"\nCleaning done!\nFinal data amount reduced to {100 * N_rows_after_cleaning / N_rows_original:2.2f} % "
        f"of original ({N_rows_after_cleaning} of {N_rows_original})"
    )

    save_local_pump_data(
        df_final,
        small_dataset=SMALL_DATASET,
        pump_number=PUMP_NUMBER,
        default_filename_addendum="_cleaned",
    )
    save_local_pump_data(
        df_discarded_rows,
        small_dataset=SMALL_DATASET,
        pump_number=PUMP_NUMBER,
        default_filename_addendum="_DISCARDED",
    )
