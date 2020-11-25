import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from src.models.AnoGen.anomaly_generator import (AnomalyGenerator,
                                                           AnomalyDetectorAutoencoder,
                                                           AnomalyEvaluator,
                                                           DetectorEvaluator)


class DataProvider:

    def __init__(self):
        self._scaler = None

    def get_local_pump_data(self, filename="train-data-large.csv", dropna=True):
        df = pd.read_csv(filename, index_col="timelocal")
        if dropna:
            df = df.dropna()
        return df

    def split_data(self, df, train_size=0.7, val_size=0.2, test_size=0.1, shuffle=False):
        x_train, x_test = train_test_split(df, test_size=val_size + test_size, shuffle=shuffle)
        x_test, x_val = train_test_split(x_test, test_size=val_size / (val_size + test_size), shuffle=shuffle)
        return x_train, x_val, x_test

    def scale_data(self, df, scaler=MinMaxScaler()):
        if self._scaler is None:
            self._scaler = scaler
        df_scaled = pd.DataFrame(self._scaler.fit_transform(df), columns=df.columns)
        return df_scaled

    def get_scaler(self):
        return self._scaler


if __name__ == "__main__":
    pd.set_option('display.max_columns', None)

    # import data
    data_handler = DataProvider()
    df = data_handler.get_local_pump_data()
    df = df.iloc[:1000, :]
    # df = df.drop("flush_indicator", axis=1)

    x_train, x_val, x_test = data_handler.split_data(df, 0.8, 0.1, 0.1)

    x_train_scaled = data_handler.scale_data(x_train)
    x_val_scaled = data_handler.scale_data(x_val)
    x_test_scaled = data_handler.scale_data(x_test)

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
    detector = AnomalyDetectorAutoencoder(intermediate_dim=12, latent_dim=2)

    # load_model = True
    load_model = False
    if load_model:
        detector.load()
    else:
        detector.fit(x_train,
                     x_val,
                     plot_history=True)
        detector.save()

    threshold = np.quantile(detector.mse_val, 0.1)
    df_samples_predicted = detector.predict(df_anomalies_decoded, threshold)

    evaluator = DetectorEvaluator()
    evaluator.plot_latent_samples_by_detection(df_vae_latent_space, df_anomalies_latent, df_samples_predicted["AE_anomaly"])


    evaluator.evaluate(df_test_nominal=x_test,
                       df_test_anomalous=df_anomalies_decoded,
                       anomaly_detector=detector,
                       # anomaly_mse_quantiles=np.linspace(0, 1, 40),
                       anomaly_thresholds=np.linspace(0, 10, 40)
                       )




    # quantile_array = np.arange(0.4, 0.6, 0.1)
    # for quantile in quantile_array:
    #     threshold = np.quantile(detector.mse_train, quantile)
    #
    #     # predict test set and synthesized anomalies
    #     df_test_anom_preds, x_test_preds = detector.predict(x_test_scaled, threshold)
    #     df_test_anom_preds.index = x_test.index
    #     df_samples_anom_preds, x_synth_preds = detector.predict(
    #         data_handler.get_scaler().transform(df_anomalies_decoded),
    #         threshold)
    #
    #     # start visualizing the predictions
    #     visualizer = AnomalyEvaluator()
    #     # visualizer.collect_visualization_dataframe(df_anomalies_latent,
    #     #                                            df_anomalies_decoded,
    #     #                                            df_samples_anom_preds,
    #     #                                            df_vae_latent_space,
    #     #                                            x_train,
    #     #                                            x_test,
    #     #                                            df_test_anom_preds,
    #     #                                            threshold,
    #     #                                            quantile)
    #
    #     visualizer.print_metrics(combine_with_test_set=True)
    #     # visualizer.plot_vae_latent(color_by_columns=["kv_flow", "leje_temp", "flush_indicator"])
    #     # visualizer.plot_vae_latent_with_samples(color_by_columns="kv_flow")
    #     visualizer.plot_vae_latent_samples_by_anomaly_prediction()
    #     # visualizer.plot_anomaly_time_series()
    print("\nEnd of test script...")
