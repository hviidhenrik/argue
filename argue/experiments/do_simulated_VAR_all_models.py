import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # silences excessive warning messages from tensorflow

from argue.models.argue import ARGUE
from argue.utils.misc import *
from argue.data.utils import *

if __name__ == "__main__":
    set_seed(1234)

    path = get_data_path() / "simulated_data"
    df_phase1 = pd.read_csv(path / f"data_simulated_VAR1_phase1.csv")
    df_phase2 = pd.read_csv(path / f"data_simulated_VAR1_phase2.csv")

    # subset data on the simulationRun (the random seed used in the TEP generation) asd
    # simulation_run = 1
    # df_phase1 = df_phase1[df_phase1["simulationRun"] == simulation_run]
    # df_phase2 = df_phase2[df_phase2["simulationRun"] == simulation_run]

    # USE_SAVED_MODEL = True
    USE_SAVED_MODEL = False
    if USE_SAVED_MODEL:
        model = ARGUE().load()
    else:
        # call and fit model
        model = ARGUE(input_dim=2,
                      number_of_decoders=2,
                      latent_dim=1)
        model.build_model(encoder_hidden_layers=[6, 5, 4, 3, 2],
                          decoders_hidden_layers=[2, 3, 4, 5, 6],
                          alarm_hidden_layers=[15, 10, 5, 3, 2],
                          gating_hidden_layers=[15, 10, 5],
                          all_activations="relu",
                          use_encoder_activations_in_alarm=True,
                          use_latent_activations_in_encoder_activations=True,
                          use_decoder_outputs_in_decoder_activations=True,
                          encoder_dropout_frac=None,
                          decoders_dropout_frac=None,
                          alarm_dropout_frac=None,
                          gating_dropout_frac=None,
                          make_model_visualiations=False
                          )
        model.fit(x_train.drop(columns=["partition"]), x_train["partition"],
                  epochs=None, autoencoder_epochs=10, alarm_gating_epochs=10,
                  batch_size=None, autoencoder_batch_size=1, alarm_gating_batch_size=1,
                  optimizer="adam",
                  ae_learning_rate=0.0001,
                  alarm_gating_learning_rate=0.0001,
                  autoencoder_decay_after_epochs=None,
                  alarm_decay_after_epochs=None,
                  gating_decay_after_epochs=None,
                  decay_rate=0.5, fp_penalty=0, fn_penalty=0,
                  validation_split=1 / 5,
                  n_noise_samples=None)
        # model.save()

    anomalies = pd.DataFrame({"x1": [0, 1, 2, -1, 4, 100, -100, 8.22],
                              "x2": [0, 1, 2, -1, 4, 100, -100, 2]})

    # predict the mixed data
    final_preds = np.round(model.predict(anomalies), 4)
    print("Alarm probabilities:\n ", np.round(model.predict_alarm_probabilities(anomalies), 4))
    print("\nGating weights:\n ", np.round(model.predict_gating_weights(anomalies), 3))
    print(f"\nFinal anomaly probabilities:\n {final_preds}")
    model.predict_plot_anomalies(anomalies, true_partitions=None)
    plt.show()

    # assert np.sum(final_preds - np.array([0.781, 0.9317, 0.9528, 0.9512, 0.9606, 0.9623, 0.9634, 0.9288])) == 0.0
