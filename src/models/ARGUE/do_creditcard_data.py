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
    path = get_data_path() / "creditcard_fraud"
    df_train = pd.read_csv(path / f"dataset_nominal_{size}.csv")
    df_test = pd.read_csv(path / f"dataset_anomalies.csv")

    # scale the data and partition it into classes
    df_train = df_train.drop(columns=["Class"])
    df_test = df_test.drop(columns=["Class"])
    scaler = MinMaxScaler().fit(df_train)
    df_train = pd.DataFrame(scaler.transform(df_train), columns=df_train.columns, index=df_train.index)
    df_test = pd.DataFrame(scaler.transform(df_test), columns=df_test.columns, index=df_test.index)
    df_train = partition_in_quantiles(df_train, "Amount", quantiles=[0, 0.5, 1])

    # Train ARGUE
    # USE_SAVED_MODEL = True
    USE_SAVED_MODEL = False
    model_path = get_model_archive_path() / "ARGUE_creditcard"
    if USE_SAVED_MODEL:
        model = ARGUE().load(model_path)
    else:
        # call and fit model
        batch_size = 256
        model = ARGUE(input_dim=len(df_train.columns[:-1]),
                      number_of_decoders=len(df_train["class"].unique()),
                      latent_dim=10, verbose=1)
        model.build_model(encoder_hidden_layers=[30, 25, 20, 15],
                          decoders_hidden_layers=[15, 20, 25, 30],
                          alarm_hidden_layers=[150, 100, 50, 20, 5],
                          gating_hidden_layers=[150, 100, 50, 20, 5],
                          all_activations="relu",
                          use_encoder_activations_in_alarm=True)
        model.fit(df_train.drop(columns=["class"]), df_train["class"],
                  epochs=None, autoencoder_epochs=500, alarm_epochs=200, gating_epochs=200,
                  batch_size=None, autoencoder_batch_size=256, alarm_gating_batch_size=256,
                  optimizer="adam",
                  autoencoder_decay_after_epochs=60,
                  alarm_gating_decay_after_epochs=20,
                  decay_rate=0.6,
                  validation_split=0.15, n_noise_samples=None, noise_stdev=1, noise_stdevs_away=3)
        model.save(model_path)

    # predict some of the training set to ensure the models are behaving correctly on this
    df_train_sanity_check = df_train.drop(columns=["class"]).sample(300).sort_index()
    model.predict_plot_reconstructions(df_train_sanity_check)
    plt.suptitle("Sanity check")
    plt.show()

    model.predict_plot_reconstructions(df_test)
    plt.suptitle("Test set")
    plt.show()

    windows_hours = list([20, 40])
    model.predict_plot_anomalies(df_train_sanity_check, window_length=windows_hours)
    plt.suptitle("Sanity check")
    plt.show()

    # predict the test set
    model.predict_plot_anomalies(df_test, window_length=windows_hours)
    plt.suptitle("Test set")
    plt.show()


