"""
This script simply initiates, fits and predicts from the FFNAutoencoder class.
Useful to test and debug.
"""

import pathlib
from sklearn.model_selection import train_test_split

from pdm.models.autoencoder.feedforward import *



if __name__ == "__main__":

    PLOTTING = True
    # PLOTTING = False
    df = pd.DataFrame(
        {
            "x1": np.sin(np.linspace(0, 10, 1000) + np.random.normal(0, 0.1, 1000)),
            "x2": np.cos(np.linspace(0, 10, 1000) + np.random.normal(0, 0.1, 1000)),
            "x3": np.cos(3.14 + np.linspace(0, 10, 1000) + np.random.normal(0, 0.1, 1000)),
        }
    )

    x_train, x_test = train_test_split(df, test_size=0.2)
    indexes = {"train_index": x_train.index, "test_index": x_test.index}

    scaler = StandardScaler().fit(x_train)
    x_train = pd.DataFrame(
        scaler.transform(x_train), columns=df.columns, index=indexes["train_index"]
    ).sort_index()
    x_test = pd.DataFrame(
        scaler.transform(x_test), columns=df.columns, index=indexes["test_index"]
    ).sort_index()
    indexes = {"train_index": x_train.index, "test_index": x_test.index}

    model = FFNAutoencoder(
        mse_quantile=0.995,
        epochs=30,
        scaler=scaler,
        hidden_layers=[5, 2, 2, 2, 5],
        activation_functions=["tanh", "tanh", "elu", "tanh", "tanh", "linear"],
        dropout_fraction=0,
        plot=PLOTTING,
        verbose=True,
    )
    plot_save_path = pathlib.Path(__file__).parents[0]
    model.fit(
        x_train=x_train,
        x_test=x_test,
        early_stopping=False,
        # plot_save_folder_path=plot_save_path,
        # latent_plot_reference_col="x3",
    )

    x_test = pd.DataFrame(
        scaler.inverse_transform(x_test), columns=df.columns, index=indexes["test_index"],
    )
    df_anomaly_results = model.predict(x_test,)  # plot_save_folder_path=plot_save_path
    df_anomaly_results["anomaly_count"] = df_anomaly_results["anomaly_autoencoder"]

    df_feature_with_anomaly_indicator = x_test.assign(
        anomaly_count=df_anomaly_results[["anomaly_count"]],
    )
    df_feature_with_anomaly_indicator = df_feature_with_anomaly_indicator.dropna()

    model.plot_reduced_timeseries(
        df_feature_with_anomaly_indicator, cols_to_display=None, pca_n_components=0
    )
