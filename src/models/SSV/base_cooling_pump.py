import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd
from pandas import DataFrame
from pdm.models.autoencoder.feedforward import FFNAutoencoder
from pdm.models.base import ModelBase, ModelParametersProvider
from pdm.models.clustering.dbscan import DBSCANClustering
from pdm.models.database.connector import ODLGoldZoneEnvironment
from pdm.models.regression.piecewise import PiecewiseLinear
from pdm.utils.definitions import filter, time_interval
from src.config.definitions import get_archive_folder_path
from src.models.SSV.filters import prep_moter_filter, prep_rpm_filter, prep_rpm_vs_motor_filter

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class CoolPumpBaseModel(ModelBase):
    """
        Base class for all cooling waterpumps predictive models.
    """

    def __init__(
        self,
        parameters: ModelParametersProvider,
        model_tags: Dict,
        model_description: str,
        tags: tuple,
        gold_zone_env: ODLGoldZoneEnvironment = ODLGoldZoneEnvironment.TEST,
    ):
        self.tags = tags
        self.parameters = parameters

        super().__init__(
            gold_zone_env,
            model_tags=dict({"product": "cooling-water-pump"}, **model_tags),
            archive_folder_path=os.path.join(get_archive_folder_path(), "cooling-water-pumps"),
            description=model_description,
        )

    def post_process_model_predict(self, df_result):
        """Return positive anomalies and anomaly count for each predicted data point

        :param df_result: a dataframe with binary anomaly indicators for each model at each time
        :type df_result: DataFrame
        :return: a dataframe with a count of anomalies and one with only data points flagged by all models
        :rtype: DataFrame
        """

        df_anomalies = df_result[
            [
                "anomaly_autoencoder",
                "anomaly_pwlf_motor_effect_kv_flow",
                "anomaly_pwlf_pump_rotation_kv_flow",
                "anomaly_pwlf_motor_effect_pump_rotation",
                "anomaly_dbscan",
            ]
        ]
        df_anomaly_results = self.replace_nulls(df_results=df_anomalies, replacement_value=-1)
        df_anomaly_results.loc[:, "ensemble_vote"] = 0
        df_anomaly_results.loc[
            (df_anomaly_results["anomaly_autoencoder"] == 1)
            | (df_anomaly_results["anomaly_pwlf_motor_effect_kv_flow"] == 1)
            | (df_anomaly_results["anomaly_pwlf_pump_rotation_kv_flow"] == 1)
            | (df_anomaly_results["anomaly_pwlf_motor_effect_pump_rotation"] == 1)
            | (df_anomaly_results["anomaly_dbscan"] == 1),
            "ensemble_vote",
        ] = 1

        df_anomaly_results = df_anomaly_results.assign(
            anomaly_count=(
                df_anomaly_results["anomaly_autoencoder"]
                + df_anomaly_results["anomaly_pwlf_motor_effect_kv_flow"]
                + df_anomaly_results["anomaly_pwlf_pump_rotation_kv_flow"]
                + df_anomaly_results["anomaly_pwlf_motor_effect_pump_rotation"]
                + df_anomaly_results["anomaly_dbscan"]
            )
        )

        df_anomaly_results.rename(
            columns={
                "anomaly_autoencoder": "ae",
                "anomaly_pwlf_motor_effect_kv_flow": "pwlf_eff_flow",
                "anomaly_pwlf_pump_rotation_kv_flow": "pwlf_rot_flow",
                "anomaly_pwlf_motor_effect_pump_rotation": "pwlf_eff_rot",
                "anomaly_dbscan": "dbscan",
            },
            inplace=True,
        )

        return df_anomaly_results

    def fit(self, df: DataFrame, verbose: bool = True):
        """
        This method is a wrapper around individual model preprocessing and fitting that executes all the models
        deemed relevant for the component in question. 
        
        Fit parameters are provided by the attribute self.parameters. 
        
        This contains a dictionary with arguments for each specific model. That is, a dictionary which holds a dictionary
        for each model, where keyworded arguments are stored as key-value pairs.

        This dictionary must contain all keyword arguments required by each model's individual fit function.
        Furthermore, it should have the following form. Arguments can be omitted, if defaults are desired.
        The dictionary must have keys 'autoencoder', 'pwlf' and 'dbscan' as depicted in the example below:

        {"autoencoder": {"epochs": 500,
                         "batch_size": 256,
                         "optimizer": Nadam(learning_rate=0.001)},

         "pwlf": {"num_linearities": 4},

         "dbscan": {"tune_k_neighbors": False,
                  "k_neighbors": 1,
                  "epsilon": 0.25,
                  "min_samples_in_cluster": 20,  # 20 is good
                  "train_split_percentage": 0.8,
                  "pca_plot": False}}

        :param df: dataframe with training data for the longest period chosen
        :type df: DataFrame
        :param verbose: should the fitting be informative or not
        :type verbose: bool, default=True
        """

        assert (
            self.model_tags["hyperparameters"] is not None
        ), "No hyperparameters have been specified"

        if verbose:
            print("=== Data ===")
            print("{0} observations with {1} features".format(*df.shape))
            print("Training period: ", df.index.min(), " to ", df.index.max())
            print("\n=== Model output ===")

        cols_autoencoder = [
            "pump_rotation",
            "motor_vibr_x",
            "motor_vibr_y",
            "leje_as_vibr_x",
            "leje_as_vibr_y",
            "leje_bs_vibr_x",
            "leje_bs_vibr_y",
            "flush_indicator",
            "leje_temp",
            "motor_leje_temp_as",
            "motor_leje_temp_bs",
            "motor_max_vikl_temp",
        ]
        cols_dbscan = ["pump_rotation", "kv_flow", "motor_effect"]

        # get model specific hyperparameters, including train period
        autoencoder_arguments = self.parameters["autoencoder"]
        pwlf_eff_flow_arguments = self.parameters["pwlf_eff_flow"]
        pwlf_rot_flow_arguments = self.parameters["pwlf_rot_flow"]
        pwlf_eff_rot_arguments = self.parameters["pwlf_eff_rot"]
        dbscan_arguments = self.parameters["dbscan"]

        # autoencoder model
        df_autoencoder = self.transform_df_for_model(
            df, autoencoder_arguments.pop("train_period", None), cols_autoencoder
        )
        x_train, x_test, scaler = self.select_and_split_and_scale_data(df_autoencoder)
        autoencoder = FFNAutoencoder(scaler=scaler, **autoencoder_arguments)
        self.model["autoencoder"] = autoencoder.fit(x_train, x_test)

        # regression models
        _mapping_model_train_period = {
            "pwlf_eff_flow": pwlf_eff_flow_arguments.pop("train_period", None),
            "pwlf_rot_flow": pwlf_rot_flow_arguments.pop("train_period", None),
            "pwlf_eff_rot": pwlf_eff_rot_arguments.pop("train_period", None),
        }
        df_eff_flow, df_rot_flow, df_eff_rot = self.select_data_for_pwlf(
            df, _mapping_model_train_period
        )

        pwlf_eff_flow = PiecewiseLinear(**pwlf_eff_flow_arguments)
        self.model["regression_effect_vs_flow"] = pwlf_eff_flow.fit(
            x=df_eff_flow["motor_effect"], y=df_eff_flow["kv_flow"]
        )

        pwlf_rot_vs_flow = PiecewiseLinear(**pwlf_rot_flow_arguments)
        self.model["regression_rotation_vs_flow"] = pwlf_rot_vs_flow.fit(
            x=df_rot_flow["pump_rotation"], y=df_rot_flow["kv_flow"]
        )

        pwlf_eff_rot = PiecewiseLinear(**pwlf_eff_rot_arguments)
        self.model["regression_effect_vs_rotation"] = pwlf_eff_rot.fit(
            x=df_eff_rot["motor_effect"], y=df_eff_rot["pump_rotation"]
        )

        # clustering model
        dbscan = DBSCANClustering(**dbscan_arguments)
        df_ready_for_clustering = self.select_data_for_dbscan(
            dbscan, cols_dbscan, df, dbscan_arguments.get("train_period")
        )
        self.model["dbscan"] = dbscan.fit(df_ready_for_clustering)

        self.serialize_model()

    def predict(self, df_feature: DataFrame, **kwargs):
        """
        Takes new data and makes anomaly predictions of it based on the trained submodels. Currently, this is
        an autoencoder neural network, a piecewise linear regression and a DBSCAN clustering model. It is simply a
        wrapper around each separate model's predict function.

        :param df_feature: DataFrame:
        :type df_feature: DataFrame:
        :return: two dataframe which the result of each minimodel type.
        :rtype: DataFrame, Dataframe
        """

        df_feature.dropna(inplace=True)

        autoencoder = self.model["autoencoder"]
        regression_effect_vs_flow = self.model["regression_effect_vs_flow"]
        regression_rotation_vs_flow = self.model["regression_rotation_vs_flow"]
        regression_effect_vs_rotation = self.model["regression_effect_vs_rotation"]
        dbscan = self.model["dbscan"]

        df_result_ae = autoencoder.predict(df_feature)

        df_eff_flow, df_rot_flow, df_eff_rot = self.select_data_for_pwlf(df_feature)
        df_result_eff_vs_flow = regression_effect_vs_flow.predict(df_eff_flow)
        df_result_rotation_vs_flow = regression_rotation_vs_flow.predict(df_rot_flow)
        df_result_eff_vs_rot = regression_effect_vs_rotation.predict(df_eff_rot)
        df_result_pwlf = pd.concat(
            [df_result_eff_vs_flow, df_result_rotation_vs_flow, df_result_eff_vs_rot], axis=1
        )

        df_result_dbscan = dbscan.predict(df_feature)
        df_result_combined = pd.concat([df_result_ae, df_result_pwlf, df_result_dbscan], axis=1)

        df_anomaly_results = self.post_process_model_predict(df_result_combined)

        if autoencoder.plot or regression_effect_vs_flow.plot or dbscan.plot:
            df_feature_with_anomaly_indicator = df_feature.assign(
                anomaly_count=df_anomaly_results[["anomaly_count"]]
            )
            self.plot_reduced_timeseries(
                df_feature_with_anomaly_indicator,
                pca_n_components=2,
                cols_to_display=["pump_rotation", "kv_flow", "leje_as_vibr_y",],
            )

        return df_anomaly_results

    def preprocess(self, df: DataFrame):
        df_preprocessed = self.remove_rows_where(df, "kv_flow", "<=", 7000)
        df_preprocessed = self.remove_rows_where(df_preprocessed, "pump_rotation", "<", 10)
        return df_preprocessed

    @staticmethod
    def apply_filter_rules(df: DataFrame, filter_rules: List):
        """
        Select given columns and filters out data based on given rules

        :param df: pivoted dataframe containing the data
        :param filter_rules: list of filter rules defined in a namedtuple
        This value will only be used if the list of values has a length of one.
        :return:
        """

        assert not df.isnull().values.any()

        for filter_rule in filter_rules:
            if filter_rule[0] is not None and filter_rule[1] is not None:
                df = df[
                    df[filter_rule[0].tag].ge(filter_rule[0].value)
                    | df[filter_rule[1].tag].le(filter_rule[1].value)
                ]
            elif filter_rule[0] is None:
                df = df[filter_rule[1].tag].le(filter_rule[1].value)
            elif filter_rule[1] is None:
                df = df[df[filter_rule[0].tag].ge(filter_rule[0].value)]
            else:
                raise ValueError

        return df

    def apply_time_and_data_filters(
        self,
        df: DataFrame,
        cols_to_include: List[str],
        data_filter_rule: List[List[filter]],
        train_period: Optional[time_interval] = None,
    ) -> DataFrame:
        """Return dataframe filtered on time period and data filters

        :param df: [description]
        :type df: DataFrame
        :param train_period: [description], defaults to None
        :type train_period: Optional[time_interval], optional
        :return: [description]
        :rtype: DataFrame
        """
        df = self.transform_df_for_model(df, train_period, cols_to_include)
        df_filtered = self.apply_filter_rules(df=df, filter_rules=data_filter_rule)

        return df_filtered

    def transform_df_for_model(
        self,
        df: DataFrame,
        time_period: Optional[time_interval] = None,
        cols_to_include: Optional[List[str]] = None,
    ) -> DataFrame:
        """Return dataframe with tags and time relevant for given model

        :param df: [description]
        :type df: DataFrame
        :param cols_to_include: [description]
        :type cols_to_include: List[str]
        :param time_period: [description]
        :type time_period: time_interval
        :return: [description]
        :rtype: DataFrame
        """
        if time_period:
            logger.info(f"selecting time period: {time_period} for columns {cols_to_include}")
            df = self.select_time_period(df, time_period)
        if cols_to_include:
            df = df[cols_to_include]
        df_transformed_with_full_row = df.dropna()

        return df_transformed_with_full_row

    def select_data_for_dbscan(
        self,
        dbscan: Any,
        cols_to_include: List[str],
        df_processed: DataFrame,
        train_period: Optional[time_interval] = None,
    ):
        df_time_selected = self.transform_df_for_model(df_processed, time_period=train_period)
        return dbscan.select_cols_for_clustering(df_time_selected, cols_to_include)

    def select_data_for_pwlf(
        self, df_processed: DataFrame, train_periods_mapping: Dict[str, time_interval] = {}
    ):
        df_eff_flow, df_rot_flow, df_eff_rot = (
            self.apply_time_and_data_filters(
                df=df_processed,
                cols_to_include=["kv_flow", "motor_effect"],
                data_filter_rule=prep_moter_filter,
                train_period=train_periods_mapping.get("pwlf_eff_flow"),
            ),
            self.apply_time_and_data_filters(
                df=df_processed,
                cols_to_include=["kv_flow", "pump_rotation"],
                data_filter_rule=prep_rpm_filter,
                train_period=train_periods_mapping.get("pwlf_rot_flow"),
            ),
            self.apply_time_and_data_filters(
                df=df_processed,
                cols_to_include=["pump_rotation", "motor_effect"],
                data_filter_rule=prep_rpm_vs_motor_filter,
                train_period=train_periods_mapping.get("pwlf_eff_rot"),
            ),
        )

        df_eff_flow = self.filter_limitpoints(df_eff_flow)
        df_rot_flow = self.filter_limitpoints(df_rot_flow)
        df_eff_rot = self.filter_limitpoints(df_eff_rot)

        return df_eff_flow, df_rot_flow, df_eff_rot
