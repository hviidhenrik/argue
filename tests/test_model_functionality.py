import numpy as np
import pandas as pd
import pytest

from argue.models.argue import ARGUE
from argue.models.argue_lite import ARGUELite
from argue.models.baseline_autoencoder import BaselineAutoencoder
from argue.utils.misc import partition_by_quantiles, set_seed

"""
Basic functionality unit tests that simply test whether the models are able to run at all, i.e. fit/train
and predict given some data.

"""


@pytest.fixture
def df_fit_data():
    # fmt: off
    return pd.DataFrame({"x1": [0.983, 0.992, 0.9976, 0.978, 0.987, 0.01, 0.003, 0.06, 0.002, 0.05],
                         "x2": [0.978, 0.988, 0.969, 0.986, 0.9975, 0.001, 0.04, 0.0031, 0.0721, 0.0034]
                         })
    # fmt: on


@pytest.fixture
def df_predict_data():
    return pd.DataFrame({"x1": [0, 1, 2, -1, 4, 100, -100, 8.22], "x2": [0, 1, 2, -1, 4, 100, -100, 2]})


def test_argue_fit_and_predict(df_fit_data, df_predict_data):
    set_seed(1234)
    x_train = partition_by_quantiles(df_fit_data, "x1", quantiles=[0, 0.5, 1])
    model = ARGUE(input_dim=2, number_of_decoders=2, latent_dim=1)
    model.build_model(
        encoder_hidden_layers=[6, 5, 4, 3, 2],
        decoders_hidden_layers=[2, 3, 4, 5, 6],
        alarm_hidden_layers=[15, 10, 5, 3, 2],
        gating_hidden_layers=[15, 10, 5],
        all_activations="relu",
    )
    model.fit(
        x_train.drop(columns=["partition"]),
        x_train["partition"],
        epochs=None,
        autoencoder_epochs=2,
        alarm_gating_epochs=2,
        batch_size=None,
        autoencoder_batch_size=1,
        alarm_gating_batch_size=1,
        validation_split=1 / 5,
        n_noise_samples=None,
    )
    final_preds_actual = np.round(model.predict(df_predict_data), 4)
    final_preds_expected = np.array([0.6572, 0.668, 0.6644, 0.6582, 0.6617, 0.6339, 0.6373, 0.5488])
    # fmt: off
    assert type(model) == ARGUE
    assert type(final_preds_actual) == np.ndarray
    assert len(final_preds_actual) == 8
    assert sum((final_preds_actual - final_preds_expected) ** 2) < 1e-02
    # fmt: on


def test_argue_lite_fit_and_predict(df_fit_data, df_predict_data):
    set_seed(1234)
    x_train = partition_by_quantiles(df_fit_data, "x1", quantiles=[0, 0.5, 1])
    model = ARGUELite(input_dim=2, latent_dim=1)
    model.build_model(
        encoder_hidden_layers=[6, 5, 4, 3, 2],
        decoders_hidden_layers=[2, 3, 4, 5, 6],
        alarm_hidden_layers=[15, 10, 5, 3, 2],
        all_activations="relu",
    )
    model.fit(
        x_train.drop(columns=["partition"]),
        epochs=None,
        autoencoder_epochs=2,
        alarm_epochs=2,
        batch_size=None,
        autoencoder_batch_size=1,
        alarm_batch_size=1,
        optimizer="adam",
        autoencoder_learning_rate=0.0001,
        alarm_learning_rate=0.0001,
        validation_split=1 / 5,
        n_noise_samples=None,
        plot_normal_vs_noise=False,
    )
    final_preds = np.round(model.predict(df_predict_data), 4)
    expected_preds = np.array([0.4818, 0.4375, 0.3942, 0.4726, 0.3652, 0.3171, 0.3735, 0.2927], dtype="float32",)

    assert type(model) == ARGUELite
    assert type(final_preds) == np.ndarray
    assert len(final_preds) == 8
    assert sum((final_preds.reshape(8) - expected_preds) ** 2) < 1e-2


def test_baseline_autoencoder_fit_and_predict(df_fit_data, df_predict_data):
    set_seed(1234)
    x_train = partition_by_quantiles(df_fit_data, "x1", quantiles=[0, 0.5, 1])
    model = BaselineAutoencoder(input_dim=2, latent_dim=1, test_set_quantile_for_threshold=0.995)
    model.build_model(
        encoder_hidden_layers=[6, 5, 4, 3, 2], decoders_hidden_layers=[2, 3, 4, 5, 6], all_activations="relu",
    )
    model.fit(
        x_train.drop(columns=["partition"]),
        epochs=1,
        batch_size=1,
        optimizer="adam",
        learning_rate=0.0001,
        validation_split=1 / 5,
    )

    final_preds = np.round(model.predict(df_predict_data), 4)
    expected_preds = np.array([1, 1, 1, 1, 1, 1, 1, 1])

    assert type(model) == BaselineAutoencoder
    assert type(final_preds) == pd.Series
    assert len(final_preds) == 8
    assert sum((final_preds - expected_preds) ** 2) < 1e-2


@pytest.mark.skip("not implemented yet")
def test_argue_lite_simultaneous_fit_and_predict():
    pass
