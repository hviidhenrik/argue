import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # silences excessive warning messages from tensorflow

from argue.models.baseline_autoencoder import BaselineAutoencoder
from argue.utils.misc import *
from argue.utils.model import *
from argue.data.utils import *

# TODO Ideas for experiments
#  - make the same models as I did for the feedwater pump model and compare performance with that using ARGUE
#  - make one big model and see if ARGUE discovers the leak in Dec 2020
#  - try different moving average windows and see if their crossings may indicate something like in finance
#  - Ideas for data partitions:
#    - cut off on mega watts, e.g. low, medium, high load
#    - do clustering using Kmeans or DBSCAN

if __name__ == "__main__":
    print("GPU: ", tf.test.is_gpu_available())
    set_seed(1234)
    # load dataset
    debugging = True
    # debugging = False
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
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(df_train)
    df_train = pd.DataFrame(scaler.transform(df_train), columns=df_train.columns, index=df_train.index)
    df_test = pd.DataFrame(scaler.transform(df_test), columns=df_test.columns, index=df_test.index)
    df_train = partition_by_quantiles(df_train, "effect_pump_30_MW", quantiles=[0, 1])

    quantiles = [0.8, 0.85, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 1.0]
    for q in quantiles:
        tf.random.set_seed(1234)
        np.random.seed(1234)

        # Train ARGUE
        # USE_SAVED_MODEL = True
        USE_SAVED_MODEL = False
        model_path = get_serialized_models_path() / "ARGUE_SSV_FWP30"
        if USE_SAVED_MODEL:
            model = BaselineAutoencoder().load(model_path)
        else:
            # call and fit model
            model = BaselineAutoencoder(input_dim=len(df_train.columns[:-1]),
                                        latent_dim=2,
                                        test_set_quantile_for_threshold=q,
                                        verbose=1)
            model.build_model(encoder_hidden_layers=[40, 35, 30, 25, 20, 15, 10, 5],
                              decoders_hidden_layers=[5, 10, 15, 20, 25, 30, 35, 40],
                              all_activations="tanh",
                              encoder_dropout_frac=None,
                              decoders_dropout_frac=None)
            model.fit(df_train.drop(columns=["partition"]),
                      epochs=1,
                      batch_size=2048,
                      optimizer="adam", learning_rate=0.001,
                      validation_split=0.1,
                      stop_early=True,
                      reduce_lr_on_plateau=True,
                      reduce_lr_by_factor=0.8,
                      noise_factor=0.0)
            # model.save(model_path)

            # predict some of the training set to ensure the models are behaving correctly on this
            df_train_sanity_check = df_train.drop(columns=["partition"]).sample(300).sort_index()
            model.predict_plot_reconstructions(df_train_sanity_check)
            plt.suptitle(f"Baseline Autoencoder Sanity check, test quantile = {q}")
            plt.savefig(get_figures_path() / f"Baseline_pump30_sanitycheck_reconstructions_q-{q}.png")
            # plt.show()

            model.predict_plot_reconstructions(df_test)
            plt.suptitle(f"Baseline Autoencoder Test set, test quantile = {q}")
            plt.savefig(get_figures_path() / f"Baseline_pump30_test_reconstructions_q-{q}.png")
            # plt.show()

            windows_hours = list(np.multiply([8, 24], 40))
            model.predict_plot_anomalies(df_train_sanity_check, window_length=windows_hours)
            plt.suptitle(f"Baseline Autoencoder Sanity check, test quantile = {q}")
            plt.savefig(get_figures_path() / f"Baseline_pump30_sanitycheck_preds_q-{q}.png")
            # plt.show()

            # predict the test set
            model.predict_plot_anomalies(df_test, window_length=windows_hours)
            plt.suptitle(f"Baseline Autoencoder Test set, test quantile = {q}")
            plt.savefig(get_figures_path() / f"Baseline_pump30_testset_preds_q-{q}.png")
            # plt.show()

            # y_pred = model.predict(df_test)
            # print("Final predictions: \n", np.round(y_pred, 3))