import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from src.models.AnoGen.anomaly_generator import (AnomalyGenerator,
                                                 AnomalyDetectorAutoencoder,
                                                 DetectorEvaluator,
                                                 DataProvider)


if __name__ == "__main__":
    pd.set_option('display.max_columns', None)

    # import data
    data_handler = DataProvider()
    df = data_handler.get_local_pump_data()
    df = df.iloc[:1000, :]
    # df = df.drop("flush_indicator", axis=1)

    x_train, x_val, x_test = data_handler.train_val_test_split(df, 0.8, 0.1, 0.1)

    x_train_scaled = data_handler.scale(x_train)
    x_val_scaled = data_handler.scale(x_val)
    x_test_scaled = data_handler.scale(x_test)

    N_anomalies = x_test.shape[0]
    generator = AnomalyGenerator(
        vae_intermediate_dim=12,
        vae_latent_dim=2)
    generator.fit(df,
                  epochs=200,
                  latent_stddev=0.005,
                  dbscan_epsilon=0.3
                  )  # TODO decide if VAE training should be on both train and test or only train set
    df_anomalies_decoded, df_anomalies_latent = generator.generate_anomalies(
        N_anomalies,
        z_min=-1,
        z_max=1,
        domain_filter={"kv_flow": [0, None],
                       "pump_rotation": [0, None]}
    )

    # plot learned latent space
    df_vae_latent_space = generator.get_vae_latent_space()
    generator.plot_vae_latent(color_by_columns=["kv_flow"], show=False)
    plt.scatter(df_anomalies_latent.iloc[:, 0], df_anomalies_latent.iloc[:, 1],
                c="black", s=10, marker="^", label="Samples")
    plt.legend()
    plt.show()

    print(df_anomalies_decoded.describe())

    # train anomaly detection algorithm
    detector = AnomalyDetectorAutoencoder(first_hidden_layer_dimension=12, latent_space_dimension=2)

    # load_model = True
    load_model = False
    if load_model:
        detector.load()
    else:
        detector.fit(x_train,
                     x_val,
                     plot_learning_curve=True)
        detector.save()

    threshold = np.quantile(detector.mse_val_set_actual_vs_predicted, 0.1)
    df_samples_predicted = detector.predict(df_anomalies_decoded, threshold)




    evaluator = DetectorEvaluator()
    evaluator.plot_latent_samples_by_detection(df_vae_latent_space, df_anomalies_latent,
                                               df_samples_predicted["anomaly"])

    evaluator.evaluate(df_test_nominal=x_test,
                       df_test_anomalous=df_anomalies_decoded,
                       anomaly_detector=detector,
                       # anomaly_mse_quantiles=np.linspace(0, 1, 40),
                       anomaly_thresholds=np.linspace(0, 10, 40)
                       )

    print("\nEnd of test script...")
