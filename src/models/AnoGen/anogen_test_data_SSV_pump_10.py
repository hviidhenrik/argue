"""This script defines the cooling water pump 10 at SSV3"""

import os
import sys
from typing import Optional

sys.path = [os.getcwd()] + sys.path


from pdm.models.base import ModelParametersProvider
from pdm.models.database.connector import ODLGoldZoneEnvironment
from pdm.utils.definitions import time_interval
from src.models.SSV import CoolPumpBaseModel

# TODO Get configs from file or use decorator
# logging.basicConfig(stream=sys.stdout,level=logging.DEBUG)


class CoolPumpModel_10(CoolPumpBaseModel):
    """Predictive maintenance model for cooling-water-pump 10 at SSV3
    """

    def __init__(
        self,
        parameters: Optional[ModelParametersProvider] = None,
        gold_zone_env: ODLGoldZoneEnvironment = ODLGoldZoneEnvironment.TEST,
    ):
        """
        :param parameters: Object holding all model hyperparameters and training period
        :type ModelTrainingParametersProvider:
        :param gold_zone_env: ODE Gold Zone environement
        :type gold_zone_env:
        """
        self.tags = (
            "SSV_03_MAG01FP101_XQ01.PV",
            "SSV_03_PAC10CE301_XQ01.PV",
            "SSV_03_PAC10GU001_XQ01.PV",
            "SSV_03_PAC10CY001_XQ01.PV",
            "SSV_03_PAC10CY002_XQ01.PV",
            "SSV_03_PAC10CY201_XQ01.PV",
            "SSV_03_PAC10CY203_XQ01.PV",
            "SSV_03_PAC10CY205_XQ01.PV",
            "SSV_03_PAC10CY207_XQ01.PV",
            "SSV_03_PAC00ED001_ZV01.PV",
            "SSV_03_PAC10CT001_XQ01.PV",
            "SSV_03_PAC10CT907_XQ01.PV",
            "SSV_03_PAC10CT908_XQ01.PV",
            "SSV_03_PAC10CT981_XQ01.PV",
        )
        self.rename = {
            "SSV_03_MAG01FP101_XQ01.PV": "kv_flow",
            "SSV_03_PAC10CE301_XQ01.PV": "motor_effect",
            "SSV_03_PAC10GU001_XQ01.PV": "pump_rotation",
            "SSV_03_PAC10CY001_XQ01.PV": "motor_vibr_x",
            "SSV_03_PAC10CY002_XQ01.PV": "motor_vibr_y",
            "SSV_03_PAC10CY201_XQ01.PV": "leje_as_vibr_x",
            "SSV_03_PAC10CY203_XQ01.PV": "leje_as_vibr_y",
            "SSV_03_PAC10CY205_XQ01.PV": "leje_bs_vibr_x",
            "SSV_03_PAC10CY207_XQ01.PV": "leje_bs_vibr_y",
            "SSV_03_PAC00ED001_ZV01.PV": "flush_indicator",
            "SSV_03_PAC10CT001_XQ01.PV": "leje_temp",
            "SSV_03_PAC10CT907_XQ01.PV": "motor_leje_temp_as",
            "SSV_03_PAC10CT908_XQ01.PV": "motor_leje_temp_bs",
            "SSV_03_PAC10CT981_XQ01.PV": "motor_max_vikl_temp",
        }
        self.model_description = """Model object for SSV3 cooling pump 10.
            Developed to detect conditions that indicate performance degradation using 
            data tags specified in 'data-tags'."""
        self.model_tags = {
            "station": "SSV3",
            "component-name": "pump10",
            "data-sources": "prodos",
            "environment": "dev",
            "gold-zone-storage": "Azure DW: bio_sp.prodos",
            "data-tags": ",".join(self.tags),
            "hyperparameters": parameters.get_params_stringified() if parameters else None,
            "training-period": parameters["full_train_period"] if parameters else None,
        }

        super().__init__(
            parameters, self.model_tags, self.model_description, self.tags, gold_zone_env
        )


if __name__ == "__main__":
    print(
        "================= Predicting from trained model for cooling water pump 10 ===============\n"
    )
    predict_period = time_interval("2020-11-01 00:00:00", "2020-11-03 23:59:00")
    pump_model = CoolPumpModel_10()
    pump_model.deserialize_model(from_model_registry=False)
    df_raw = pump_model.fetch_and_widen_data(predict_period)
    # df_raw = pump_model.get_local_test_data(train=False, large_dataset=True)
    # pump_model.save_local_test_data(df_raw, train=False, large_dataset=True)
    df_ready_to_predict = pump_model.preprocess(df_raw)  # consider drop nan
    pump_model.predict_and_load_results_to_dw(df_ready_to_predict, save_to_dw=False)

    print("\n\n================ End of script ================")
