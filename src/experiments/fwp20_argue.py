import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # silences excessive warning messages from tensorflow

from src.models.argue import ARGUE
from src.utils.misc import *
from src.data.utils import *

if __name__ == "__main__":
    set_seed(1234)

    path = get_data_path() / "ssv_feedwater_pump"
    figure_path = get_figures_path() / "ssv_feedwater_pump" / "pump_20"

    # get phase 1 and 2 data
    df_train = get_local_data(path / f"data_pump20_phase1.csv")
    df_train_meta = df_train[["sample", "faulty"]]
    df_train = df_train.drop(columns=["sample", "faulty"])

    df_test = get_local_data(path / f"data_pump20_phase2.csv")
    df_test_meta = df_test[["sample", "faulty"]]
    df_test = df_test.drop(columns=["sample", "faulty"])

    # scale the data
    scaler = MinMaxScaler().fit(df_train)
    df_train = pd.DataFrame(scaler.transform(df_train), columns=df_train.columns, index=df_train.index)
    df_test = pd.DataFrame(scaler.transform(df_test), columns=df_test.columns, index=df_test.index)

    # visualize principal components to discover natural clusters in the data and use those for partitioning
    df_pca_data, pca_object = reduce_dimension_by_pca(df_train)
    # plot_candidate_partitions_by_pca(df_pca_data, pca_object)
    partition_labels = select_pcs_and_partition_data(df_pca_data, pcs_to_cluster_on=[2, 3], n_clusters=2,
                                                     plot_pca_clustering=False)
    df_train["partition"] = partition_labels

    # Train ARGUE model
    # USE_SAVED_MODEL = True
    USE_SAVED_MODEL = False
    model_path = get_serialized_models_path() / "ARGUE_SSV_FWP20"
    if USE_SAVED_MODEL:
        model = ARGUE().load(model_path)
    else:
        model = ARGUE(input_dim=len(df_train.columns[:-1]),
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
                  epochs=None, autoencoder_epochs=20, alarm_gating_epochs=10,
                  batch_size=None, autoencoder_batch_size=2048, alarm_gating_batch_size=2048,
                  optimizer="adam", ae_learning_rate=0.001, alarm_gating_learning_rate=0.001,
                  autoencoder_decay_after_epochs=80,
                  alarm_decay_after_epochs=60,
                  gating_decay_after_epochs=None,
                  decay_rate=0.5,
                  validation_split=0.1,
                  n_noise_samples=None, noise_stdev=1, noise_stdevs_away=4)
        model.save(model_path)

    # predict some of the training set to ensure the models are behaving correctly on this
    df_train_sanity_check = df_train.drop(columns=["partition"]).sample(300).sort_index()
    model.predict_plot_reconstructions(df_train_sanity_check)
    plt.suptitle("ARGUE Sanity check")
    # plt.savefig(figure_path / f"ARGUE_pump20_sanitycheck_reconstructions.png")
    plt.show()

    model.predict_plot_reconstructions(df_test)
    plt.suptitle("ARGUE Test set")
    # plt.savefig(figure_path / f"ARGUE_pump20_test_reconstructions.png")
    plt.show()

    windows_hours = list(np.multiply([8, 24], 40))
    model.predict_plot_anomalies(df_train_sanity_check, window_length=windows_hours)
    plt.suptitle("ARGUE Sanity check")
    # plt.savefig(figure_path / f"ARGUE_pump20_sanitycheck_preds.png")
    plt.show()

    # predict the test set
    model.predict_plot_anomalies(df_test, window_length=windows_hours)
    plt.suptitle("ARGUE Test set")
    plt.savefig(figure_path / f"ARGUE_pump20_testset_preds.png")
    # plt.show()