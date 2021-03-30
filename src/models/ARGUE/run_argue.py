import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from src.models.ARGUE.models import ARGUE
from src.models.ARGUE.data_generation import *

if __name__ == "__main__":
    # make some data
    N = 10000
    df = pd.concat([pd.DataFrame(np.random.normal(loc=[100, 3, 10],
                                                  scale=[15, 1, 1],
                                                  size=(N, 3)), ),
                    pd.DataFrame(np.random.exponential(scale=[10, 3, 10],
                                                       size=(N, 3)), ),
                    pd.DataFrame(np.random.uniform(low=[0, 0, 0],
                                                   high=[4, 3, 4],
                                                   size=(N, 3)), )
                    ]).reset_index(drop=True)
    df.columns = ["x1", "x2", "x3"]
    df["class"] = make_class_labels(3, N)
    # df.plot(subplots=True)
    # plt.show()

    # scale it
    scaler = StandardScaler()
    df[["x1", "x2", "x3"]] = scaler.fit_transform(df[["x1", "x2", "x3"]])

    # call and fit model
    model = ARGUE(input_dim=3,
                  number_of_decoders=3,
                  latent_dim=6)
    model.build_model(encoder_hidden_layers=[64, 48, 32, 16, 8],
                      decoders_hidden_layers=[8, 16, 32, 48, 64],
                      alarm_hidden_layers=[64, 48, 32, 16, 8],
                      gating_hidden_layers=[64, 48, 32, 16, 8],
                      all_activations="tanh")
    model.fit(df.drop(columns=["class"]), df["class"], epochs=30, number_of_batches=32, batch_size=256,
              verbose=1, n_noise_samples=N, optimizer="adam")

    model.save()

    # make new data which contains some normal and anomalous samples
    healthy_indices = [0, N - 1, 2 * N - 1]
    normal_samples = df[["x1", "x2", "x3"]].iloc[healthy_indices]
    healthy_samples = pd.DataFrame(np.random.normal(loc=[100, 3, 10],
                                                    scale=[15, 1, 1],
                                                    size=(10, 3)),
                                   columns=["x1", "x2", "x3"])
    healthy_samples = pd.DataFrame(scaler.transform(healthy_samples), columns=healthy_samples.columns)

    # predict the mixed data
    test_samples = pd.concat([normal_samples, healthy_samples]).reset_index(drop=True)
    print("Alarm probabilities: ", model.predict_alarm_probabilities(test_samples))
    print("\nGating weights: ", model.predict_gating_weights(test_samples))
    print("\nFinal anomaly probabilities: ", model.predict(test_samples))

    # model_loaded = ARGUE().load()
