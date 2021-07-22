import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # silences excessive warning messages from tensorflow

from src.models.baseline_autoencoder import BaselineAutoencoder
from src.utils.experiment_logger import ExperimentLogger
from src.utils.misc import *
from src.data.utils import *

if __name__ == "__main__":
    set_seed(1234)

    path = get_data_path() / "ssv_feedwater_pump"
    figure_path = get_figures_path() / "ssv_feedwater_pump" / "pump_30"

    # get phase 1 and 2 data
    df_train = get_local_data(path / f"data_pump30_phase1.csv")
    df_train_meta = df_train[["sample", "faulty"]]
    df_train = df_train.drop(columns=["sample", "faulty"])

    df_test = get_local_data(path / f"data_pump30_phase2.csv")
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

    quantiles = [0.8, 0.85, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 1.0]
    quantiles = [0.995]
    for q in quantiles:
        set_seed(1234)
        # Train ARGUE
        # USE_SAVED_MODEL = True
        USE_SAVED_MODEL = False
        model_path = get_serialized_models_path() / "BaselineAutoencoder_SSV_FWP30"
        if USE_SAVED_MODEL:
            model = BaselineAutoencoder().load(model_path)
        else:
            # call and fit model
            model = BaselineAutoencoder(input_dim=len(df_train.columns[:-1]),
                                        latent_dim=5,
                                        test_set_quantile_for_threshold=q,
                                        verbose=1)
            model.build_model(encoder_hidden_layers=[40, 35, 30, 25, 20, 15, 10, 5],
                              decoders_hidden_layers=[5, 10, 15, 20, 25, 30, 35, 40],
                              all_activations="tanh",
                              encoder_dropout_frac=None,
                              decoders_dropout_frac=None)
            model.fit(df_train.drop(columns=["partition"]),
                      epochs=200,
                      batch_size=2048,
                      optimizer="adam", learning_rate=0.001,
                      validation_split=0.1,
                      stop_early=True,
                      reduce_lr_on_plateau=True,
                      reduce_lr_by_factor=0.8,
                      noise_factor=0.0)
            # model.save(model_path)

        # save hyperparameters and other model info to csv
        logger = ExperimentLogger()
        logger.save_model_parameter_log(model, "fwp30_baselineAE")
        exp_id = logger.get_experiment_id()

        # predict some of the training set to ensure the models are behaving correctly on this
        df_train_sanity_check = df_train.drop(columns=["partition"]).sample(300).sort_index()
        model.predict_plot_reconstructions(df_train_sanity_check)
        plt.suptitle(f"Baseline Autoencoder Sanity check, test quantile = {q}")
        # plt.savefig(figure_path / f"Baseline_pump30_sanitycheck_reconstructions_q-{q}_ID{exp_id}.png")
        plt.show()

        # get the exact time where the fault starts
        idx_fault_start = df_test_meta.index[np.where(df_test_meta["faulty"] == 1)[0][0]]

        model.predict_plot_reconstructions(df_test)
        plt.suptitle(f"Baseline Autoencoder Test set, test quantile = {q}")
        # plt.savefig(figure_path / f"Baseline_pump30_test_reconstructions_q-{q}_ID{exp_id}.png")
        plt.show()

        windows_hours = list(np.multiply([8, 24], 40))
        model.predict_plot_anomalies(df_train_sanity_check, window_length=windows_hours)
        plt.suptitle(f"Baseline Autoencoder Sanity check, test quantile = {q}")
        # plt.savefig(figure_path / f"Baseline_pump30_sanitycheck_preds_q-{q}_ID{exp_id}.png")
        plt.show()

        # predict the test set
        model.predict_plot_anomalies(df_test, window_length=windows_hours)
        plt.vlines(x=idx_fault_start, ymin=0, ymax=1, color="red")
        plt.suptitle(f"Baseline Autoencoder Test set, test quantile = {q}")
        plt.savefig(figure_path / f"Baseline_pump30_testset_preds_q-{q}_ID{exp_id}.png")
        # plt.show()
