import os

import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # silences excessive warning messages from tensorflow

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from src.models.ARGUE.models import ARGUE
from src.models.ARGUE.data_generation import *
from src.models.ARGUE.utils import *
from src.data.data_utils import *

if __name__ == "__main__":
    tf.random.set_seed(1234)
    np.random.seed(1234)
    # make some data
    x_train = pd.DataFrame({"x1": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                            "x2": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]})
    x_train = partition_by_quantiles(x_train, "x1", quantiles=[0, 0.5, 1])

    # USE_SAVED_MODEL = True
    USE_SAVED_MODEL = False
    if USE_SAVED_MODEL:
        model = ARGUE().load()
    else:
        # call and fit model
        model = ARGUE(input_dim=2,
                      number_of_decoders=2,
                      latent_dim=1)
        model.build_model(encoder_hidden_layers=[1, 1],
                          decoders_hidden_layers=[1, 1],
                          alarm_hidden_layers=[1, 1],
                          gating_hidden_layers=[1, 1],
                          all_activations="relu",
                          use_encoder_activations_in_alarm=False,
                          use_latent_activations_in_encoder_activations=False,
                          use_decoder_outputs_in_decoder_activations=False,
                          encoder_dropout_frac=None,
                          decoders_dropout_frac=None,
                          alarm_dropout_frac=None,
                          gating_dropout_frac=None,
                          make_model_visualiations=False
                          )
        model.fit(x_train.drop(columns=["partition"]), x_train["partition"],
                  epochs=None, autoencoder_epochs=0, alarm_gating_epochs=2,
                  batch_size=None, autoencoder_batch_size=1, alarm_gating_batch_size=1,
                  optimizer="adam", ae_learning_rate=0.1, alarm_gating_learning_rate=0.1,
                  autoencoder_decay_after_epochs=None,
                  alarm_decay_after_epochs=None,
                  gating_decay_after_epochs=None,
                  decay_rate=0.5, fp_penalty=0, fn_penalty=0,
                  validation_split=1/6,
                  n_noise_samples=None, noise_stdev=1, noise_stdevs_away=10)
        # model.save(model_path)

    # make new data which contains some normal and anomalous samples
    healthy_samples = make_custom_test_data(5, 5, 5, noise_sd=noise_sds)
    healthy_labels = make_class_labels(3, 5)
    healthy_samples.plot(subplots=True)
    plt.show()

    anomalies = pd.DataFrame(np.array([
        [50, 50, 50],
        [200, 200, 200],
        [-30, -30, -30],
    ]).reshape(-1, 3), columns=healthy_samples.columns)
    anomaly_labels = [-1 for _ in range(anomalies.shape[0])]
    test_samples = pd.concat([healthy_samples, anomalies]).reset_index(drop=True)
    test_samples = pd.DataFrame(scaler.transform(test_samples), columns=test_samples.columns)

    model.predict_plot_reconstructions(test_samples)
    plt.show()

    # predict the mixed data
    print("Alarm probabilities:\n ", model.predict_alarm_probabilities(test_samples))
    print("\nGating weights:\n ", model.predict_gating_weights(test_samples))
    print(f"\nFinal anomaly probabilities:\n {np.round(model.predict(test_samples), 4)}")
    model.predict_plot_anomalies(test_samples, true_classes=healthy_labels + anomaly_labels)
    plt.show()
