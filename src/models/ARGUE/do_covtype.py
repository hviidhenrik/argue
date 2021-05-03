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

# TODO covtype:
#  - look into why all alarm probs can be very low near zero (shouldnt be possible)
#  - also, gating network seems to only choose virtual decision on small test set (debug)
#       - something is going on with the labeling or shuffling inside .fit, probably
#       - also, could be a problem with the predict function, since the training seems fine, but the predictions
#         achieve a very low AUC and accuracy at 50% == random. Look into .predict method and trace the problem back


if __name__ == "__main__":


    # load dataset
    debugging = True
    # debugging = False
    size = get_dataset_purpose_as_str(debugging)
    covtype = "A"
    # covtype = "B"
    path = get_data_path() / "covtype"
    x_train = pd.read_csv(path / f"data_covtype_normal_{covtype}_{size}.csv", index_col=False)
    x_test_anomaly = pd.read_csv(path / f"data_covtype_anomaly_{covtype}_{size}.csv")
    x_train, x_test_normal = train_test_split(x_train, test_size=x_test_anomaly.shape[0])

    # scale the data and partition it into partitions
    x_test_normal["anomaly"] = 0
    x_test_anomaly["anomaly"] = 1
    x_test = pd.concat([x_test_normal, x_test_anomaly]).reset_index(drop=True)

    # partition by feature classes
    y_test = x_test["anomaly"]
    x_test = x_test.drop(columns=["anomaly"])
    _, x_test_debug, _, y_test_debug = train_test_split(x_test, y_test, test_size=1)

    x_test_debug_partitions = x_test_debug["Cover_Type"]
    x_test = x_test.drop(columns=["Cover_Type"])
    x_test_debug = x_test_debug.drop(columns=["Cover_Type"])
    x_train_partitions = x_train["Cover_Type"]
    x_train = x_train.drop(columns=["Cover_Type"])

    scaler = MinMaxScaler().fit(x_train)
    x_train = pd.DataFrame(scaler.transform(x_train), columns=x_train.columns, index=x_train.index)
    x_test = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns, index=x_test.index)
    x_test_debug = pd.DataFrame(scaler.transform(x_test_debug), columns=x_test_debug.columns, index=x_test_debug.index)

    # Train ARGUE
    # USE_SAVED_MODEL = True
    USE_SAVED_MODEL = False
    model_path = get_model_archive_path() / "ARGUE_covtype"
    if USE_SAVED_MODEL:
        model = ARGUE().load(model_path)
    else:
        # call and fit model
        model = ARGUE(input_dim=len(x_train.columns),
                      number_of_decoders=len(x_train_partitions.unique()),
                      latent_dim=15, verbose=1)
        model.build_model(encoder_hidden_layers=[90, 75, 60, 45, 25, 15],
                          decoders_hidden_layers=[15, 25, 45, 60, 75, 90],
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
        model.fit(x_train, x_train_partitions,
                  epochs=None, autoencoder_epochs=1, alarm_gating_epochs=1,
                  batch_size=None, autoencoder_batch_size=128, alarm_gating_batch_size=1024,
                  optimizer="adam",
                  autoencoder_decay_after_epochs=None,
                  alarm_decay_after_epochs=None,
                  gating_decay_after_epochs=None,
                  decay_rate=0.7, fp_penalty=0, fn_penalty=0,
                  validation_split=0.15,
                  n_noise_samples=None, noise_stdev=1, noise_stdevs_away=3)
        model.save(model_path)

    # predict some of the training set to ensure the models are behaving correctly on this
    # x_train_sanity_check = x_train.drop(columns=["Cover_Type"]).sample(300).sort_index()
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
    print("Alarm probs: \n", np.round(alarm, 2))
    print("Gating weights: \n", np.round(gating, 2))
    print("Final predictions: \n", np.round(y_pred, 2))

    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    print("\n >> ARGUE Performance evaluation: ")
    print(f"Bin acc: {accuracy_score(y_test, np.round(y_pred)):.4f}")
    print(f"AP:      {average_precision_score(y_test, y_pred):.4f}")
    print(f"PR AUC:  {auc(recall, precision):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_pred):.4f}")

    # for debugging
    y_pred = model.predict(x_test_debug)
    alarm = model.predict_alarm_probabilities(x_test_debug)
    gating = model.predict_gating_weights(x_test_debug)
    print("Alarm probs: \n", np.round(alarm, 2))
    print("Gating weights: \n", np.round(gating, 2))
    print("True partitions: \n", np.array(x_test_debug_partitions))
    print("Final predictions: \n", np.round(y_pred, 2))
    print("True alarm labels: \n", np.array(y_test_debug))
