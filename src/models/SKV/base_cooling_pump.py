import logging
from logging import log
import os
from typing import Any, Dict, List, Optional

import pandas as pd
from pandas import DataFrame
from pdm.models.base import ModelBase, ModelParametersProvider
from pdm.models.autoencoder.feedforward import FFNAutoencoder
from pdm.models.clustering.dbscan import DBSCANClustering
from pdm.models.database.connector import ODLGoldZoneEnvironment
from pdm.models.regression.piecewise import PiecewiseLinear
from pdm.utils.definitions import filter, time_interval
from src.config.definitions import get_archive_folder_path
from src.models.SKV.filters import prep_moter_filter, prep_rpm_filter, prep_rpm_vs_motor_filter

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

        expected_columns_set = set([
                "anomaly_pwlf_motor_effect_active_kv_flow",
                "anomaly_pwlf_pump_rotation_active_kv_flow",
                "anomaly_pwlf_motor_effect_active_pump_rotation_active",
                "anomaly_pwlf_motor_effect_kv_flow",
                "anomaly_pwlf_pump_rotation_kv_flow",
                "anomaly_pwlf_motor_effect_pump_rotation",
                "anomaly_dbscan",
                "anomaly_autoencoder",
            ])
        
        actual_columns_set = set(list(df_result.columns))
        expected_columns_not_present = list(expected_columns_set - actual_columns_set)

        for col in expected_columns_not_present:
            df_result[col] = 0.0

        df_anomalies = df_result[expected_columns_set]
        df_anomaly_results = self.replace_nulls(df_results=df_anomalies, replacement_value=-1)
        df_anomaly_results.loc[:, "ensemble_vote"] = 0
        df_anomaly_results.loc[
            (df_anomaly_results["anomaly_pwlf_motor_effect_active_kv_flow"] == 1)
            | (df_anomaly_results["anomaly_pwlf_pump_rotation_active_kv_flow"] == 1)
            | (df_anomaly_results["anomaly_pwlf_motor_effect_active_pump_rotation_active"] == 1)
            | (df_anomaly_results["anomaly_pwlf_motor_effect_kv_flow"] == 1)
            | (df_anomaly_results["anomaly_pwlf_pump_rotation_kv_flow"] == 1)
            | (df_anomaly_results["anomaly_pwlf_motor_effect_pump_rotation"] == 1)
            | (df_anomaly_results["anomaly_dbscan"] == 1)
            | (df_anomaly_results["anomaly_autoencoder"] == 1),
            "ensemble_vote",
        ] = 1

        df_anomaly_results = df_anomaly_results.assign(
            anomaly_count=(
                df_anomaly_results["anomaly_pwlf_motor_effect_active_kv_flow"]
                + df_anomaly_results["anomaly_pwlf_pump_rotation_active_kv_flow"]
                + df_anomaly_results["anomaly_pwlf_motor_effect_active_pump_rotation_active"]
                + df_anomaly_results["anomaly_pwlf_motor_effect_kv_flow"]
                + df_anomaly_results["anomaly_pwlf_pump_rotation_kv_flow"]
                + df_anomaly_results["anomaly_pwlf_motor_effect_pump_rotation"]
                + df_anomaly_results["anomaly_dbscan"]
                + df_anomaly_results["anomaly_autoencoder"]
            )
        )

        df_anomaly_results.rename(
            columns={
                "anomaly_pwlf_motor_effect_active_kv_flow": "pwlf_eff_flow_one",
                "anomaly_pwlf_pump_rotation_active_kv_flow": "pwlf_rot_flow_one",
                "anomaly_pwlf_motor_effect_active_pump_rotation_active": "pwlf_eff_rot_one",
                "anomaly_pwlf_motor_effect_kv_flow": "pwlf_eff_flow",
                "anomaly_pwlf_pump_rotation_kv_flow": "pwlf_rot_flow",
                "anomaly_pwlf_motor_effect_pump_rotation": "pwlf_eff_rot",
                "anomaly_dbscan": "dbscan",
                "anomaly_autoencoder": "ae",
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
            "pump_rotation_other",
            "motor_effect",
            "motor_effect_other",
            "kv_flow",
        ]

        cols_dbscan = [
            "pump_rotation",
            "pump_rotation_other",
            "motor_effect",
            "motor_effect_other",
            "kv_flow",
        ]

        # get model specific hyperparameters, including train period
        autoencoder_arguments = self.parameters["autoencoder"]
        pwlf_eff_flow_arguments_one = self.parameters["pwlf_eff_flow_one"]
        pwlf_rot_flow_arguments_one = self.parameters["pwlf_rot_flow_one"]
        pwlf_eff_rot_arguments_one = self.parameters["pwlf_eff_rot_one"]
        pwlf_eff_flow_arguments_both = self.parameters["pwlf_eff_flow_both"]
        pwlf_rot_flow_arguments_both = self.parameters["pwlf_rot_flow_both"]
        pwlf_eff_rot_arguments_both = self.parameters["pwlf_eff_rot_both"]
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
            "autoencoder": autoencoder_arguments.pop("train_period", None),
            "pwlf_eff_flow_one": pwlf_eff_flow_arguments_one.pop("train_period", None),
            "pwlf_rot_flow_one": pwlf_rot_flow_arguments_one.pop("train_period", None),
            "pwlf_eff_rot_one": pwlf_eff_rot_arguments_one.pop("train_period", None),
            "pwlf_eff_flow_both": pwlf_eff_flow_arguments_both.pop("train_period", None),
            "pwlf_rot_flow_both": pwlf_rot_flow_arguments_both.pop("train_period", None),
            "pwlf_eff_rot_both": pwlf_eff_rot_arguments_both.pop("train_period", None),
        }

        _rpm_min = 100
        _rpm_not_running_cutoff = 1

        # first select rows where ONLY ONE pump is active and then
        # remove any pump rotation values for both pumps in the interval (_rpm_not_running_cutoff, _rpm_min)
        df_one_active = df[(df["pump_rotation"] > _rpm_min)
                           ^ (df["pump_rotation_other"] > _rpm_min)]
        df_one_active = df_one_active[
            ((df_one_active["pump_rotation"] < _rpm_not_running_cutoff)
             | (df_one_active["pump_rotation"] > _rpm_min))
            & (
                (df_one_active["pump_rotation_other"] < _rpm_not_running_cutoff)
                | (df_one_active["pump_rotation_other"] > _rpm_min)
            )
        ]
        df_one_active["pump_rotation_active"] = df_one_active[
            ["pump_rotation", "pump_rotation_other"]
        ].max(axis=1)
        df_one_active["motor_effect_active"] = df_one_active[
            ["motor_effect", "motor_effect_other"]
        ].max(axis=1)

        # select only rows where BOTH pumps were running
        df_both_active = df[(df["pump_rotation"] > _rpm_min) & (df["pump_rotation_other"] > _rpm_min)]

        # pwlf - only one pump running
        pwlf_eff_flow_one = PiecewiseLinear(**pwlf_eff_flow_arguments_one)
        self.model["regression_effect_vs_flow_one"] = pwlf_eff_flow_one.fit(
            x=df_one_active["motor_effect_active"], y=df_one_active["kv_flow"]
        )

        pwlf_rot_vs_flow_one = PiecewiseLinear(**pwlf_rot_flow_arguments_one)
        self.model["regression_rotation_vs_flow_one"] = pwlf_rot_vs_flow_one.fit(
            x=df_one_active["pump_rotation_active"], y=df_one_active["kv_flow"]
        )

        pwlf_eff_rot_one = PiecewiseLinear(**pwlf_eff_rot_arguments_one)
        self.model["regression_effect_vs_rotation_one"] = pwlf_eff_rot_one.fit(
            x=df_one_active["motor_effect_active"], y=df_one_active["pump_rotation_active"]
        )

        # pwlf - both pumps running
        pwlf_eff_flow_both = PiecewiseLinear(**pwlf_eff_flow_arguments_both)
        self.model["regression_effect_vs_flow_both"] = pwlf_eff_flow_both.fit(
            x=df_both_active["motor_effect"], y=df_both_active["kv_flow"]
        )

        pwlf_rot_vs_flow_both = PiecewiseLinear(**pwlf_rot_flow_arguments_both)
        self.model["regression_rotation_vs_flow_both"] = pwlf_rot_vs_flow_both.fit(
            x=df_both_active["pump_rotation"], y=df_both_active["kv_flow"]
        )

        pwlf_eff_rot_both = PiecewiseLinear(**pwlf_eff_rot_arguments_both)
        self.model["regression_effect_vs_rotation_both"] = pwlf_eff_rot_both.fit(
            x=df_both_active["motor_effect"], y=df_both_active["pump_rotation"]
        )

        # clustering model
        df_clustering = df[df["pump_rotation"] > _rpm_min]
        dbscan = DBSCANClustering(**dbscan_arguments)
        df_ready_for_clustering = self.select_data_for_dbscan(
            dbscan, cols_dbscan, df_clustering, dbscan_arguments.get("train_period")
        )
        self.model["dbscan"] = dbscan.fit(df_ready_for_clustering)

        self.serialize_model()

    def _is_dataframe_valid(self, df_validator_expression: Any) -> bool:
        return not (df_validator_expression)
    
    def _concat_dataframes(self, df_result: DataFrame, *dfs_to_add) -> DataFrame:     
        _list_with_df_to_concat = [df_result] + [df for df in dfs_to_add]
        
        return pd.concat(_list_with_df_to_concat, axis=1)

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

        if not self._is_dataframe_valid((df_feature.empty | df_feature.shape[0] < 30)):
            logger.info(
                f"Feature dataframe for model {self.name} for station {self.model_tags['station']}" 
                " is lacking enough data to perform a prediction. Terminating execution."
            )
            return (
                pd.DataFrame() # TODO returning empty df, as model base expects a df to be returned
            )  

        autoencoder = self.model["autoencoder"]
        regression_effect_vs_flow_one = self.model["regression_effect_vs_flow_one"]
        regression_rotation_vs_flow_one = self.model["regression_rotation_vs_flow_one"]
        regression_effect_vs_rotation_one = self.model["regression_effect_vs_rotation_one"]
        regression_effect_vs_flow_both = self.model["regression_effect_vs_flow_both"]
        regression_rotation_vs_flow_both = self.model["regression_rotation_vs_flow_both"]
        regression_effect_vs_rotation_both = self.model["regression_effect_vs_rotation_both"]
        dbscan = self.model["dbscan"]

        df_result_ae = autoencoder.predict(df_feature)

        _rpm_min = 100
        _rpm_not_running_cutoff = 1

        # select only rows where ONE pump was running
        df_one_active = df_feature[
            (df_feature["pump_rotation"] > _rpm_min)
            ^ (df_feature["pump_rotation_other"] > _rpm_min)
        ]
        df_one_active = df_one_active[
            ((df_one_active["pump_rotation"] < _rpm_not_running_cutoff)
             | (df_one_active["pump_rotation"] > _rpm_min))
            & (
                (df_one_active["pump_rotation_other"] < _rpm_not_running_cutoff)
                | (df_one_active["pump_rotation_other"] > _rpm_min)
            )
        ]
        df_one_active["pump_rotation_active"] = df_one_active[
            ["pump_rotation", "pump_rotation_other"]
        ].max(axis=1)
        df_one_active["motor_effect_active"] = df_one_active[
            ["motor_effect", "motor_effect_other"]
        ].max(axis=1)

        # select only rows where BOTH pumps were running
        df_both_active = df_feature[
            (df_feature["pump_rotation"] > _rpm_min)
            & (df_feature["pump_rotation_other"] > _rpm_min)
        ]

        df_result_pwlf = pd.DataFrame()

        if self._is_dataframe_valid((df_one_active.empty | df_one_active.shape[0] < 30)):
            df_result_eff_vs_flow_one = regression_effect_vs_flow_one.predict(df_one_active)
            df_result_rotation_vs_flow_one = regression_rotation_vs_flow_one.predict(df_one_active)
            df_result_eff_vs_rot_one = regression_effect_vs_rotation_one.predict(df_one_active)

            df_result_pwlf = self._concat_dataframes(
                df_result_pwlf,
                df_result_eff_vs_flow_one,
                df_result_rotation_vs_flow_one,
                df_result_eff_vs_rot_one
                )

        if self._is_dataframe_valid((df_both_active.empty | df_both_active.shape[0] < 30)):
            df_result_eff_vs_flow_both = regression_effect_vs_flow_both.predict(df_both_active)
            df_result_rotation_vs_flow_both = regression_rotation_vs_flow_both.predict(df_both_active)
            df_result_eff_vs_rot_both = regression_effect_vs_rotation_both.predict(df_both_active)

            df_result_pwlf = self._concat_dataframes(
                df_result_pwlf,
                df_result_eff_vs_flow_both,
                df_result_rotation_vs_flow_both,
                df_result_eff_vs_rot_both
                )

        #df_result_pwlf = pd.concat(
        #    [
        #        df_result_eff_vs_flow_one,
        #        df_result_rotation_vs_flow_one,
        #        df_result_eff_vs_rot_one,
        #        df_result_eff_vs_flow_both,
        #        df_result_rotation_vs_flow_both,
        #        df_result_eff_vs_rot_both,
        #    ],
        #    axis=1,
        #)

        df_clustering = df_feature[df_feature["pump_rotation"] > _rpm_min]
        df_result_dbscan = dbscan.predict(df_clustering)
        df_result_combined = pd.concat([df_result_ae, df_result_pwlf, df_result_dbscan], axis=1)

        df_anomaly_results = self.post_process_model_predict(df_result_combined)

        if any([model_object[1].plot for model_object in self.model.items()]):
            df_feature_with_anomaly_indicator = df_feature.assign(
                anomaly_count=df_anomaly_results[["anomaly_count"]]
            )
            self.plot_reduced_timeseries(
                df_feature_with_anomaly_indicator,
                pca_n_components=2,
                cols_to_display=["pump_rotation", "kv_flow", "motor_effect"],
            )

        return df_anomaly_results

    def preprocess(self, df: DataFrame):
        df_preprocessed = self.remove_rows_where(df, "kv_flow", "<=", 3)
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
