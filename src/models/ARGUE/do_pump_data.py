import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # silences excessive warning messages from tensorflow
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from src.models.ARGUE.models import ARGUE
from src.models.ARGUE.data_generation import *
from src.models.ARGUE.utils import *
from src.config.definitions import *
from src.data.data_utils import *

# TODO Ideas for experiments
#  - make the same models as I did for the feedwater pump model and compare performance with that using ARGUE
#  - make one big model and see if ARGUE discovers the leak in Dec 2020
#  - Ideas for data partitions:
#    - cut off on mega watts, e.g. low, medium, high load
#    - do clustering using Kmeans or DBSCAN
if __name__ == "__main__":
    # load dataset
    # debugging = True
    debugging = False
    size = get_dataset_purpose_as_str(debugging)
    path = get_data_path() / "ssv_feedwater_pump" / f"data_pump_30_{size}_cleaned.csv"
    df_raw = get_local_data(path)

    # form train and test sets
    df_train = df_raw.loc[:"2020-09-01 23:59:59"]
    df_test = df_raw.loc["2020-09-02":]

    # scale the data and partition it into classes
    scaler = StandardScaler().fit(df_train)
    df_train = pd.DataFrame(scaler.transform(df_train), columns=df_train.columns, index=df_train.index)
    df_test = pd.DataFrame(scaler.transform(df_test), columns=df_test.columns, index=df_test.index)
    df_train = partition_in_quantiles(df_train, "effect_pump_30_MW")

    # Train ARGUE
    # USE_SAVED_MODEL = True
    USE_SAVED_MODEL = False
    model_path = get_model_archive_path() / "ARGUE_SSV_FWP30"
    if USE_SAVED_MODEL:
        model = ARGUE().load(model_path)
    else:
        # call and fit model
        model = ARGUE(input_dim=len(df_train.columns[:-1]),
                      number_of_decoders=len(df_train["class"].unique()),
                      latent_dim=10)
        model.build_model(encoder_hidden_layers=[25, 20, 15],
                          decoders_hidden_layers=[15, 20, 25],
                          alarm_hidden_layers=[60, 50, 40, 30, 20, 10],
                          gating_hidden_layers=[60, 50, 40, 30, 20, 10],
                          all_activations="tanh")
        model.fit(df_train.drop(columns=["class"]), df_train["class"],
                  epochs=3, autoencoder_epochs=40, alarm_gating_epochs=5,
                  autoencoder_batch_size=256, alarm_gating_batch_size=256,
                  optimizer="adam", validation_split=0.2, noise_mean=4, noise_stdev=1)
        model.save(model_path)

    model.predict_plot_reconstructions(df_test)
    plt.show()

    # predict and plot the mixed data
    model.predict_plot_anomalies(df_test, df_test.index, moving_average_window=120)
    plt.show()


