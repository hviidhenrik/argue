import os
import sys

sys.path = [os.getcwd()] + sys.path

from keras.optimizers import Nadam
from pdm.models.base import ModelParametersProvider
from pdm.utils.definitions import time_interval
from src.models.SSV.pump_20 import CoolPumpModel_20

print("================= Training model for cooling water pump 20 ===================\n")
model_parameters = ModelParametersProvider(
    train_period=time_interval("2020-09-20 17:00:00", "2020-09-22 15:00:00")
)
# dbscan
model_parameters["autoencoder"] = {
    "epochs": 200,
    "batch_size": 128,
    "optimizer": Nadam(learning_rate=0.001),
    "verbose": True,
    "plot": True,
}
# regression
model_parameters["pwlf_eff_flow"] = {
    "num_linearities": 5,
    "conf_level": 0.9999,
    "plot": True,
    "verbose": True,
}
model_parameters["pwlf_rot_flow"] = {
    "num_linearities": 5,
    "conf_level": 0.9999,
    "plot": True,
    "verbose": True,
}
model_parameters["pwlf_eff_rot"] = {
    "num_linearities": 5,
    "conf_level": 0.9999,
    "plot": True,
    "verbose": True,
}
# dbscan
model_parameters["dbscan"] = {
    "tune_k_neighbors": False,
    "k_neighbors": 1,
    "epsilon": 0.25,
    "min_samples_in_cluster": 20,
    "train_split_percentage": 0.9,
    "verbose": True,
    "plot": True,
    "pca_plot": False,
}

pump_model = CoolPumpModel_20(model_parameters)
df_raw_full_period = pump_model.fetch_and_widen_data(model_parameters["full_train_period"])
df_preproc_full_period = pump_model.preprocess(df_raw_full_period)
pump_model.fit(df_preproc_full_period, verbose=True)

print("Trained and serialized model locally")

##### Register - uncomment below lines to register model to model registry #####
# registred_amls_model = pump_model.serialize_model(to_model_registry=True)
# print("Successfully registered {} to AMLS workspace".format(pump_model.name))
