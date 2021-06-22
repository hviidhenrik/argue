import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # silences excessive warning messages from tensorflow

from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from src.models.ARGUE.argue import ARGUE
from src.models.ARGUE.utils import *
from src.data.data_utils import *

# TODO Ideas for experiments
#  - make the same models as I did for the feedwater pump model and compare performance with that using ARGUE
#  - make one big model and see if ARGUE discovers the leak in Dec 2020
#  - try different moving average windows and see if their crossings may indicate something like in finance
#  - Ideas for data partitions:
#    - cut off on mega watts, e.g. low, medium, high load
#    - do clustering using Kmeans or DBSCAN

if __name__ == "__main__":
    tf.random.set_seed(1234)
    np.random.seed(1234)
    # load dataset
    # debugging = True
    debugging = False
    size = get_dataset_purpose_as_str(debugging)
    path = get_data_path() / "ssv_feedwater_pump" / f"data_pump_30_{size}_cleaned.csv"
    df_raw = get_local_data(path)
    df_raw = df_raw[["effect_pump_30_MW", "flow_after_pump", "temp_after_pump", "temp_slipring_water_suction_side",
                     "temp_slipring_water_pressure_side", "temp_slipring_diff"]]
    # form train and test sets
    df_train = pd.concat([df_raw.loc[:"2019-12-01 23:59:59"],
                          df_raw.loc["2020-02-30 23:59:59":
                                     "2020-09-14 23:59:59"]])
    # df_test = get_df_with_bad_data(df_train, df_raw)
    df_test = df_raw.loc["2020-09-15":]
    # df_test.plot(subplots=True, rot=5)
    # plt.suptitle("SSV Feedwater pump 30 temperature tags")
    # plt.show()

    # scale the data and partition it into classes
    scaler = MinMaxScaler().fit(df_train)
    df_train = pd.DataFrame(scaler.transform(df_train), columns=df_train.columns, index=df_train.index)
    df_test = pd.DataFrame(scaler.transform(df_test), columns=df_test.columns, index=df_test.index)
    # TODO try standard ARGUE with a clustering model on the data instead of quantile partitions
    # df_train = partition_by_quantiles(df_train, "effect_pump_30_MW", quantiles=[0, 0.5, 1])
    df_train = partition_by_pca_and_clustering(df_train, 2, 2)


    # Train ARGUE
    # USE_SAVED_MODEL = True
    USE_SAVED_MODEL = False
    model_path = get_model_archive_path() / "ARGUE_SSV_FWP30"
    if USE_SAVED_MODEL:
        model = ARGUE().load(model_path)
    else:
        # call and fit model
        model = ARGUE(input_dim=len(df_train.columns[:-1]),  # TODO revise input dims with partition etc to be easier
                      number_of_decoders=len(df_train["partition"].unique()),
                      latent_dim=2, verbose=1)
        model.build_model(encoder_hidden_layers=[40, 35, 30, 25, 20, 15, 10, 5],
                          decoders_hidden_layers=[5, 10, 15, 20, 25, 30, 35, 40],
                          alarm_hidden_layers=[320, 160, 80, 40, 20, 10],
                          all_activations="tanh",
                          use_encoder_activations_in_alarm=True,
                          use_latent_activations_in_encoder_activations=True,
                          use_decoder_outputs_in_decoder_activations=True,
                          encoder_dropout_frac=None,
                          decoders_dropout_frac=None,
                          alarm_dropout_frac=None,
                          gating_dropout_frac=None)
        model.fit(df_train.drop(columns=["partition"]), df_train["partition"],
                  epochs=None, autoencoder_epochs=200, alarm_gating_epochs=20,
                  batch_size=None, autoencoder_batch_size=2048, alarm_gating_batch_size=2048,
                  optimizer="adam", ae_learning_rate=0.001, alarm_gating_learning_rate=0.001,
                  autoencoder_decay_after_epochs=80,
                  alarm_decay_after_epochs=60,
                  gating_decay_after_epochs=None,
                  decay_rate=0.5,
                  validation_split=0.1,
                  n_noise_samples=None, noise_stdev=1, noise_stdevs_away=4)
        # model.save(model_path)

    # predict some of the training set to ensure the models are behaving correctly on this
    df_train_sanity_check = df_train.drop(columns=["partition"]).sample(300).sort_index()
    model.predict_plot_reconstructions(df_train_sanity_check)
    plt.suptitle("ARGUE Sanity check")
    plt.savefig(get_ARGUE_path() / "plots" / f"ARGUE_pump30_sanitycheck_reconstructions.png")
    # plt.show()

    model.predict_plot_reconstructions(df_test)
    plt.suptitle("ARGUE Test set")
    plt.savefig(get_ARGUE_path() / "plots" / f"ARGUE_pump30_test_reconstructions.png")
    # plt.show()

    windows_hours = list(np.multiply([8, 24], 40))
    model.predict_plot_anomalies(df_train_sanity_check, window_length=windows_hours)
    plt.suptitle("ARGUE Sanity check")
    plt.savefig(get_ARGUE_path() / "plots" / f"ARGUE_pump30_sanitycheck_preds.png")
    # plt.show()

    # predict the test set
    model.predict_plot_anomalies(df_test, window_length=windows_hours)
    plt.suptitle("ARGUE Test set")
    plt.savefig(get_ARGUE_path() / "plots" / f"ARGUE_pump30_testset_preds.png")
    # plt.show()

    # y_pred = model.predict(df_test)
    # alarm = model.predict_alarm_probabilities(df_test)
    #
    # gating = model.predict_gating_weights(df_test)
    # print("Alarm probs: \n", np.round(alarm, 3))
    # print("Gating weights: \n", np.round(gating, 3))
    # print("Final predictions: \n", np.round(y_pred, 3))
