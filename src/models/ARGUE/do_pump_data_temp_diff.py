import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # silences excessive warning messages from tensorflow
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from src.models.ARGUE.models import ARGUE
from src.models.ARGUE.data_generation import *
from src.models.ARGUE.utils import *
from src.config.definitions import *
from src.data.data_utils import *

# TODO Ideas for experiments
#  - make the same models as I did for the feedwater pump model and compare performance with that using ARGUE
#  - make one big model and see if ARGUE discovers the leak in Dec 2020
#  - try different moving average windows and see if their crossings may indicate something like in finance
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
    df_raw = df_raw[["effect_pump_30_MW", "flow_after_pump", "temp_after_pump", "temp_slipring_water_suction_side",
                     "temp_slipring_water_pressure_side", "temp_slipring_diff"]]
    # form train and test sets
    df_train = pd.concat([df_raw.loc[:"2019-12-30 23:59:59"],
                          df_raw.loc["2020-02-05 23:59:59":"2020-09-14 23:59:59"]])
    df_test = df_raw.loc["2020-09-15":]
    df_test.plot(subplots=True, rot=5)
    plt.suptitle("SSV Feedwater pump 30 temperature tags")
    plt.show()

    # scale the data and partition it into classes
    scaler = MinMaxScaler().fit(df_train)
    df_train = pd.DataFrame(scaler.transform(df_train), columns=df_train.columns, index=df_train.index)
    df_test = pd.DataFrame(scaler.transform(df_test), columns=df_test.columns, index=df_test.index)
    df_train = partition_in_quantiles(df_train, "effect_pump_30_MW", quantiles=[0, 0.2, 0.3, 0.4, 0.5, 0.6,
                                                                                0.7, 0.8, 0.9, 1])

    # Train ARGUE
    # USE_SAVED_MODEL = True
    USE_SAVED_MODEL = False
    model_path = get_model_archive_path() / "ARGUE_SSV_FWP30"
    if USE_SAVED_MODEL:
        model = ARGUE().load(model_path)
    else:
        # call and fit model
        batch_size = 256
        model = ARGUE(input_dim=len(df_train.columns[:-1]),
                      number_of_decoders=len(df_train["partition"].unique()),
                      latent_dim=2, verbose=1)
        model.build_model(encoder_hidden_layers=[15, 10, 5, 4, 3],
                          decoders_hidden_layers=[3, 4, 5, 10, 15],
                          alarm_hidden_layers=[40, 30, 20, 10],
                          gating_hidden_layers=[20, 10, 5],
                          all_activations="relu",
                          use_encoder_activations_in_alarm=True)
        model.fit(df_train.drop(columns=["partition"]), df_train["partition"],
                  epochs=None, autoencoder_epochs=300, alarm_gating_epochs=100,
                  batch_size=None, autoencoder_batch_size=256, alarm_gating_batch_size=256,
                  optimizer="adam",
                  autoencoder_decay_after_epochs=None,
                  alarm_gating_decay_after_epochs=None,
                  decay_rate=0.7, fp_penalty=0, fn_penalty=0,
                  validation_split=0.15,
                  n_noise_samples=None, noise_stdev=1, noise_stdevs_away=3)
        # model.save(model_path)

    # predict some of the training set to ensure the models are behaving correctly on this
    df_train_sanity_check = df_train.drop(columns=["partition"]).sample(300).sort_index()
    model.predict_plot_reconstructions(df_train_sanity_check)
    plt.suptitle("Sanity check")
    plt.show()

    model.predict_plot_reconstructions(df_test)
    plt.suptitle("Test set")
    plt.show()

    windows_hours = list(np.multiply([8, 24], 40))
    model.predict_plot_anomalies(df_train_sanity_check, window_length=windows_hours)
    plt.suptitle("Sanity check")
    plt.show()

    # predict the test set
    model.predict_plot_anomalies(df_test, window_length=windows_hours)
    plt.suptitle("Test set")
    plt.show()

    y_pred = model.predict(df_test)
    alarm = model.predict_alarm_probabilities(df_test)
    gating = model.predict_gating_weights(df_test)
    print("Alarm probs: \n", np.round(alarm, 4))
    print("Gating weights: \n", np.round(gating, 4))
    print("Final predictions: \n", np.round(y_pred, 4))
