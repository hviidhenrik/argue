import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # silences excessive warning messages from tensorflow
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from src.models.ARGUE.models import ARGUE
from src.models.ARGUE.data_generation import *

if __name__ == "__main__":
    # make some data
    N = 10000
    noise_sds = [10, 30, 6]
    noise_sds = [0, 0, 0]
    df = make_custom_test_data(N, N, N, noise_sd=noise_sds)
    df.columns = ["x1", "x2", "x3"]
    df["class"] = make_class_labels(classes=3, N=N)

    import tensorflow as tf
    foo = tf.keras.optimizers.Adam()

    # scale it
    scaler = StandardScaler()
    df[["x1", "x2", "x3"]] = scaler.fit_transform(df[["x1", "x2", "x3"]])
    df.plot(subplots=True)
    plt.show()

    # USE_SAVED_MODEL = True
    USE_SAVED_MODEL = False
    if USE_SAVED_MODEL:
        model = ARGUE().load()
    else:
        # call and fit model
        model = ARGUE(input_dim=3,
                      number_of_decoders=3,
                      latent_dim=5)
        model.build_model(encoder_hidden_layers=[10, 8, 7],
                          decoders_hidden_layers=[7, 8, 10],
                          alarm_hidden_layers=[15, 10, 5, 3],
                          gating_hidden_layers=[15, 12, 10],
                          all_activations="tanh")
        model.fit(df.drop(columns=["class"]), df["class"], epochs=7, number_of_batches=32, batch_size=256,
                  verbose=1, n_noise_samples=N, optimizer="adam", validation_split=0.2,
                  noise_mean=50, noise_sd=0.5)
        # model.save()

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
