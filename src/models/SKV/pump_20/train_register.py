import os
import sys

sys.path = [os.getcwd()] + sys.path

from keras.optimizers import Nadam
from pdm.models.base import ModelParametersProvider
from pdm.utils.definitions import time_interval
from src.models.SKV.pump_20 import CoolPumpModel_20

print("================= Training model for cooling water pump 20 ===================\n")
model_parameters = ModelParametersProvider(
    train_period=time_interval("2020-02-10 00:00:00", "2020-06-01 23:59:00")
)

# plotting = True
plotting = False

# autoencoder
model_parameters["autoencoder"] = {
    "epochs": 200,
    "batch_size": 128,
    "optimizer": Nadam(learning_rate=0.001),
    "verbose": True,
    "plot": plotting,
}

# regression
model_parameters["pwlf_eff_flow_one"] = {
    "num_linearities": 2,
    "conf_level": 0.9999,
    "plot": plotting,
    "verbose": True,
}
model_parameters["pwlf_rot_flow_one"] = {
    "num_linearities": 2,
    "conf_level": 0.9999,
    "plot": plotting,
    "verbose": True,
}
model_parameters["pwlf_eff_rot_one"] = {
    "num_linearities": 2,
    "conf_level": 0.9999,
    "plot": plotting,
    "verbose": True,
}
model_parameters["pwlf_eff_flow_both"] = {
    "num_linearities": 2,
    "conf_level": 0.9999,
    "plot": plotting,
    "verbose": True,
}
model_parameters["pwlf_rot_flow_both"] = {
    "num_linearities": 2,
    "conf_level": 0.9999,
    "plot": plotting,
    "verbose": True,
}
model_parameters["pwlf_eff_rot_both"] = {
    "num_linearities": 2,
    "conf_level": 0.9999,
    "plot": plotting,
    "verbose": True,
}

# dbscan
model_parameters["dbscan"] = {
    "k_neighbors": 1,
    "epsilon": 0.7,
    "min_samples_in_cluster": 20,
    "verbose": True,
    "plot": plotting,
}

pump_model = CoolPumpModel_20(model_parameters)
df_raw_full_period = pump_model.fetch_and_widen_data(model_parameters["full_train_period"])
# df_raw_full_period = pump_model.get_local_test_data(train=True, large_dataset=True)
# pump_model.save_local_test_data(df_raw_full_period, train=True, large_dataset=True)
df_preproc_full_period = pump_model.preprocess(df_raw_full_period)
pump_model.fit(df_preproc_full_period, verbose=True)

print("Trained and serialized model locally")

##### Register - uncomment below lines to register model to model registry #####
#registred_amls_model = pump_model.serialize_model(to_model_registry=True)
#print("Successfully registered {} to AMLS workspace".format(pump_model.name))
