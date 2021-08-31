"""
This script cleans the data from feedwater pump 30 at SSV
"""
import logging
import sys
from src.data.ssv_fwp_data_utility_functions import *
from src.data.utils import *

PUMP_NUMBER = "30"
# SMALL_DATASET = True
SMALL_DATASET = False
SAVE_PLOTS = False
USE_SAVED_KDE = False

if len(sys.argv) > 1:
    SMALL_DATASET = get_dataset_size(sys.argv[1])

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if __name__ == "__main__":
    plt.style.use("seaborn")
    PUMP_OTHER_NUMBER = get_other_pump_number(PUMP_NUMBER)
    DATASET_SIZE = get_dataset_size(SMALL_DATASET)

    # --------------------------
    logger.info(f"===== Feedwater pump {PUMP_NUMBER} data cleaning script =====")

    # load data
    path = get_data_path() / "ssv_feedwater_pump"
    df = pd.read_csv(path / f"data_pump_{PUMP_NUMBER}_{DATASET_SIZE}.csv", index_col="timelocal")
    N_rows_original = df.shape[0]
    df = df.dropna().drop_duplicates()

    logger.info(f"Raw data period: {df.index.min()} to {df.index.max()}")

    # choose initial data period
    df = df.loc["2020-09-15": "2020-12-31"]
    df = make_feature_engineered_tags(df)

    plot_effect_vs_flow(df, PUMP_NUMBER, "RAW DATA", save=SAVE_PLOTS)
    plt.show()

    df = apply_SME_bounds_filter(df, PUMP_NUMBER, megawatt_lower=1.1, megawatt_upper=10,
                                 flow_lower_bound=38, flow_upper_bound=200)
    logger.info(f"Training data period: {df.index.min()} to {df.index.max()}")

    # set apriori known limits for normal operation cf. Rasmus and Jonas
    df_not_cleaned = df.copy()

    # pump 20 effect has some outliers as high as 70 - like 30 tho, it can only run at max 10 MW
    df = df.drop(
        df[
            (df[f"effect_pump_20_MW"] > 10)
        ].index
    )
    plot_effect_vs_flow(
        df, PUMP_NUMBER, "AFTER BOUNDS FILTER", save=SAVE_PLOTS
    )

    # define linear cut to sift away some bad data
    plt.plot(df[f"effect_pump_30_MW"], 18 * df[f"effect_pump_30_MW"] + 3)
    plt.show()

    N_rows_after_cleaning = df.shape[0]
    plot_effect_vs_flow(
        df, PUMP_NUMBER, "AFTER BOUNDS FILTER", save=SAVE_PLOTS
    )
    plt.show()

    plot_slipring_temperature_difference(df, pump_number=PUMP_NUMBER, save=SAVE_PLOTS)
    plt.show()

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

    plot_effect_vs_flow(
        df_this_pump_active, PUMP_NUMBER, f"ONLY {PUMP_NUMBER} ACTIVE", save=SAVE_PLOTS
    )
    plt.show()

    plot_effect_vs_flow(
        df_both_active, PUMP_NUMBER, "BOTH ACTIVE", save=SAVE_PLOTS
    )
    plt.plot(df_both_active[f"effect_pump_30_MW"], 18 * df_both_active[f"effect_pump_30_MW"] + 70)
    plt.show()
    # df_both_active = filter_points_beyond_linear_cut(df_both_active, x_col=f"effect_pump_{PUMP_NUMBER}_MW",
    #                                                  y_col="flow_after_pump", slope=18, intercept=70,
    #                                                  remove_below_cut=False)

    kde_filename_this_pump_active = (
        f"KDE_pump_{PUMP_NUMBER}_active_{DATASET_SIZE}_dataset.pkl"
    )

    df_normal_this = df_this_pump_active
    df_normal_both = df_both_active

    plot_effect_vs_flow(
        df_normal_this,
        PUMP_NUMBER,
        f"AFTER KDE CLEANING\nONLY PUMP {PUMP_NUMBER} ACTIVE",
        save=SAVE_PLOTS,
    )
    plt.show()

    plot_effect_vs_flow(
        df_normal_both, PUMP_NUMBER, "AFTER KDE CLEANING\nBOTH ACTIVE", save=SAVE_PLOTS
    )
    plt.show()

    # inspect temperature difference in water between sliprings (may indicate leak if very large)
    plot_slipring_temperature_difference(
        df_normal_this,
        pump_number=PUMP_NUMBER,
        title_addendum=f"ONLY {PUMP_NUMBER} ACTIVE",
        save=SAVE_PLOTS,
    )
    plt.show()

    plot_slipring_temperature_difference(
        df_normal_both,
        pump_number=PUMP_NUMBER,
        title_addendum="BOTH ACTIVE",
        save=SAVE_PLOTS,
    )
    plt.show()

    # do a final cleaning of the biggest outliers in each column
    logger.info("Filtering data on each column...")
    # df_normal_this = filter_data_on_tail_quantiles(
    #     df_normal_this, upper_quantile=0.995, lower_quantile=0.005
    # )
    # df_normal_both = filter_data_on_tail_quantiles(
    #     df_normal_both, upper_quantile=0.995, lower_quantile=0.005
    # )

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
        plt.show()

    for column in [
        f"effect_pump_{PUMP_NUMBER}_MW",
        "flow_after_pump",
        "temp_slipring_diff",
    ]:
        plot_column_as_timeseries(
            df_discarded_rows[column],
            column,
            PUMP_NUMBER,
            "DISCARDED DATA",
            save=SAVE_PLOTS,
        )
        plt.show()

    N_rows_after_cleaning = df_final.shape[0]
    logger.info(
        f"\nCleaning done!\nFinal data amount reduced to {100 * N_rows_after_cleaning / N_rows_original:2.2f} % "
        f"of original ({N_rows_after_cleaning} of {N_rows_original})"
    )

    df_final = df_final.drop(columns=["effect_pump_10_MW"])

    # make columns with sample number and binary fault status as the first columns in the df
    df_final = make_sample_and_faulty_cols(df_final)
    df_final.loc["2020-11-14 20:00:00":"2021-01-08", "faulty"] = 1
    save_local_data(df_final, filename=path / "data_pump30_phase2.csv", index="timelocal")
