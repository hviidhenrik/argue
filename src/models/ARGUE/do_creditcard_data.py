import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # silences excessive warning messages from tensorflow
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (confusion_matrix, roc_auc_score, precision_recall_curve, auc, average_precision_score,
                             accuracy_score)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from src.models.ARGUE.models import ARGUE
from src.models.ARGUE.data_generation import *
from src.models.ARGUE.utils import *
from src.config.definitions import *
from src.data.data_utils import *

# TODO Ideas for experiments
#  - Ideas for data partitions:
#    - do clustering using Kmeans or DBSCAN

if __name__ == "__main__":
    # load dataset
    # debugging = True
    debugging = False
    size = get_dataset_purpose_as_str(debugging)
    path = get_data_path() / "creditcard_fraud"
    x_train = pd.read_csv(path / f"dataset_nominal_{size}.csv")
    x_test_anomaly = pd.read_csv(path / f"dataset_anomalies.csv")

    # scale the data and partition it into partitions
    x_train, x_test_normal = train_test_split(x_train, test_size=x_test_anomaly.shape[0])
    x_train = x_train.drop(columns=["Class"])
    x_test = pd.concat([x_test_normal, x_test_anomaly]).reset_index(drop=True)
    y_test = x_test["Class"]
    x_test = x_test.drop(columns=["Class"])

    scaler = MinMaxScaler().fit(x_train)
    x_train = pd.DataFrame(scaler.transform(x_train), columns=x_train.columns, index=x_train.index)
    x_test = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns, index=x_test.index)
    x_train = partition_by_quantiles(x_train, "Amount", quantiles=[0, 0.5, 1])

    # Train ARGUE
    # USE_SAVED_MODEL = True
    USE_SAVED_MODEL = False
    model_path = get_model_archive_path() / "ARGUE_creditcard"
    if USE_SAVED_MODEL:
        model = ARGUE().load(model_path)
    else:
        # call and fit model
        model = ARGUE(input_dim=len(x_train.columns[:-1]),  # -1 here because we dont want the "partition" column
                      number_of_decoders=len(x_train["partition"].unique()),
                      latent_dim=5, verbose=1)
        model.build_model(encoder_hidden_layers=[50, 40, 30, 20, 15],
                          decoders_hidden_layers=[15, 20, 30, 40, 50],
                          alarm_hidden_layers=[1000, 500, 200, 75],
                          gating_hidden_layers=[1000, 500, 200, 75],
                          all_activations="relu",
                          use_encoder_activations_in_alarm=True,
                          use_latent_activations_in_encoder_activations=True,
                          use_decoder_outputs_in_decoder_activations=True,
                          encoder_dropout_frac=0.1,
                          decoders_dropout_frac=0.1,
                          alarm_dropout_frac=0.1,
                          gating_dropout_frac=0.1
                          )
        model.fit(x_train.drop(columns=["partition"]), x_train["partition"],
                  epochs=None, autoencoder_epochs=100, alarm_gating_epochs=100,
                  batch_size=None, autoencoder_batch_size=256, alarm_gating_batch_size=256,
                  optimizer="adam",
                  autoencoder_decay_after_epochs=None,
                  alarm_gating_decay_after_epochs=None,
                  decay_rate=0.5, fp_penalty=0, fn_penalty=0,
                  validation_split=0.15,
                  n_noise_samples=None, noise_stdev=1, noise_stdevs_away=10)
        # model.save(model_path)

    # predict some of the training set to ensure the models are behaving correctly on this
    x_train_sanity_check = x_train.drop(columns=["partition"]).sample(300).sort_index()
    # model.predict_plot_reconstructions(x_train_sanity_check)
    # plt.suptitle("Sanity check")
    # plt.show()
    #
    # model.predict_plot_reconstructions(x_test)
    # plt.suptitle("Test set")
    # plt.show()

    # evaluate model performance on test set
    y_pred = model.predict(x_test)
    alarm = model.predict_alarm_probabilities(x_test)
    gating = model.predict_gating_weights(x_test)
    print("Alarm probs: \n", np.round(alarm, 4))
    print("Gating weights: \n", np.round(gating, 4))

    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    print("\n >> ARGUE Performance evaluation: ")
    print(f"Bin acc: {accuracy_score(y_test, np.round(y_pred)):.4f}")
    print(f"AP:      {average_precision_score(y_test, y_pred):.4f}")
    print(f"PR AUC:  {auc(recall, precision):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_pred):.4f}")

    # for debugging
    _, x_test, _, y_test = train_test_split(x_test, y_test, test_size=10)
    y_pred = model.predict(x_test)
    alarm = model.predict_alarm_probabilities(x_test)
    gating = model.predict_gating_weights(x_test)
    print("Alarm probs: \n", np.round(alarm, 4))
    print("Gating weights: \n", np.round(gating, 4))
    print("Final predictions: \n", np.round(y_pred, 4))
    print("True alarm labels: \n", np.round(y_test, 4))
