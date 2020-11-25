"""This script defines the cooling water pump 10 at SKV3"""

import os
import sys
from typing import Optional

sys.path = [os.getcwd()] + sys.path


from pdm.models.base import ModelParametersProvider
from pdm.models.database.connector import ODLGoldZoneEnvironment
from pdm.utils.definitions import time_interval
from src.models.SKV import CoolPumpBaseModel

# TODO Get configs from file or use decorator
# logging.basicConfig(stream=sys.stdout,level=logging.DEBUG)


class CoolPumpModel_10(CoolPumpBaseModel):
    """Predictive maintenance model for cooling-water-pump 10 at SKV3
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
            "SKV_03_PAC10FU650_XQ01.PV",
            "SKV_03_PAC20FU650_XQ01.PV",  # "pump 20"
            "SKV_03_PAC10FU650_XQ02.PV",
            "SKV_03_PAC10GU001D_XQ01.PV",
            "SKV_03_PAC20GU001D_XQ01.PV",  # pump 20
            # "SKV_03_PAC10FU650_XQ03.PV"
        )

        self.rename = {
            "SKV_03_PAC10FU650_XQ01.PV": "motor_effect",
            "SKV_03_PAC20FU650_XQ01.PV": "motor_effect_other",  # "pump 20"
            "SKV_03_PAC10FU650_XQ02.PV": "kv_flow",
            "SKV_03_PAC10GU001D_XQ01.PV": "pump_rotation",
            "SKV_03_PAC20GU001D_XQ01.PV": "pump_rotation_other",  # "pump 20"
            # "SKV_03_PAC10FU650_XQ03.PV": "hydraulic_effect"
        }
        self.model_description = """Model object for SKV3 cooling pump 10.
            Developed to detect conditions that indicate performance degradation using 
            data tags specified in 'data-tags'."""
        self.model_tags = {
            "station": "SKV3",
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
    predict_period = time_interval("2020-10-02 00:00:00", "2020-10-14 10:00:00")
    pump_model = CoolPumpModel_10()
    pump_model.deserialize_model(from_model_registry=False)
    df_raw = pump_model.fetch_and_widen_data(predict_period)
    # df_raw = pump_model.get_local_test_data(train=False, large_dataset=True)
    # pump_model.save_local_test_data(df_raw, train=False, large_dataset=True)
    df_ready_to_predict = pump_model.preprocess(df_raw)
    pump_model.update_model_tags({"inference_requester": "pipeline"})
    pump_model.predict_and_load_results_to_dw(df_ready_to_predict, save_to_dw=True)

    print("\n\n================ End of script ================")