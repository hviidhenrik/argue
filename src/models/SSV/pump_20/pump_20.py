"""This script defines the cooling water pump 20 at SSV3"""

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


class CoolPumpModel_20(CoolPumpBaseModel):
    """
        this Class performs predictive maintenance on coolwater pump 20 at SSV
    """

    def __init__(
        self,
        parameters: Optional[ModelParametersProvider] = None,
        gold_zone_env: ODLGoldZoneEnvironment = ODLGoldZoneEnvironment.TEST,
    ):
        """

        :param train_period:
        :type train_period:
        :param gold_zone_env:
        :type gold_zone_env:
        """
        self.tags = (
            "SSV_03_MAG01FP101_XQ01.PV",
            "SSV_03_PAC20CE301_XQ01.PV",
            "SSV_03_PAC20GU001_XQ01.PV",
            "SSV_03_PAC20CY001_XQ01.PV",
            "SSV_03_PAC20CY002_XQ01.PV",
            "SSV_03_PAC20CY201_XQ01.PV",
            "SSV_03_PAC20CY203_XQ01.PV",
            "SSV_03_PAC20CY205_XQ01.PV",
            "SSV_03_PAC20CY207_XQ01.PV",
            "SSV_03_PAC00ED001_ZV01.PV",
            "SSV_03_PAC20CT001_XQ01.PV",
            "SSV_03_PAC20CT907_XQ01.PV",
            "SSV_03_PAC20CT908_XQ01.PV",
            "SSV_03_PAC20CT981_XQ01.PV",
        )
        self.rename = {
            "SSV_03_MAG01FP101_XQ01.PV": "kv_flow",
            "SSV_03_PAC20CE301_XQ01.PV": "motor_effect",
            "SSV_03_PAC20GU001_XQ01.PV": "pump_rotation",
            "SSV_03_PAC20CY001_XQ01.PV": "motor_vibr_x",
            "SSV_03_PAC20CY002_XQ01.PV": "motor_vibr_y",
            "SSV_03_PAC20CY201_XQ01.PV": "leje_as_vibr_x",
            "SSV_03_PAC20CY203_XQ01.PV": "leje_as_vibr_y",
            "SSV_03_PAC20CY205_XQ01.PV": "leje_bs_vibr_x",
            "SSV_03_PAC20CY207_XQ01.PV": "leje_bs_vibr_y",
            "SSV_03_PAC00ED001_ZV01.PV": "flush_indicator",
            "SSV_03_PAC20CT001_XQ01.PV": "leje_temp",
            "SSV_03_PAC20CT907_XQ01.PV": "motor_leje_temp_as",
            "SSV_03_PAC20CT908_XQ01.PV": "motor_leje_temp_bs",
            "SSV_03_PAC20CT981_XQ01.PV": "motor_max_vikl_temp",
        }
        self.model_description = """Model object for SSV3 cooling pump 20.
            Developed to detect conditions that indicate performance degradation using 
            motor, flow, rpm and vibration data"""
        self.model_tags = {
            "station": "SSV3",
            "component-name": "pump20",
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
        "================= Predicting from trained model for cooling water pump 20 ===============\n"
    )
    predict_period = time_interval("2020-09-20 17:00:00", "2020-09-22 03:30:00")
    pump_model = CoolPumpModel_20()
    pump_model.deserialize_model(from_model_registry=False)
    df_raw = pump_model.fetch_and_widen_data(predict_period)
    df_ready_to_predict = pump_model.preprocess(df_raw)
    pump_model.predict_and_load_results_to_dw(df_ready_to_predict, save_to_dw=True)

    print("\n\n================ End of script ================")
