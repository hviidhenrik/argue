import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from argue.models.argue_lite import ARGUELite
from argue.utils.misc import partition_by_quantiles, set_seed

if __name__ == "__main__":
    set_seed(1234)
    # make some data
    # fmt: off
    x_train = pd.DataFrame(
        {
            "x1": [0.983, 0.992, 0.9976, 0.978, 0.987, 0.01, 0.003, 0.06, 0.002, 0.05],
            "x2": [0.978, 0.988, 0.969, 0.986, 0.9975, 0.001, 0.04, 0.0031, 0.0721, 0.0034],
        }
    )
    x_train = partition_by_quantiles(x_train, "x1", quantiles=[0, 1])
    # fmt: on

    # USE_SAVED_MODEL = True
    USE_SAVED_MODEL = False
    if USE_SAVED_MODEL:
        model = ARGUELite().load()
    else:
        # call and fit model
        model = ARGUELite(input_dim=2, latent_dim=1)
        model.build_model(
            encoder_hidden_layers=[6, 5, 4, 3, 2],
            decoders_hidden_layers=[2, 3, 4, 5, 6],
            alarm_hidden_layers=[15, 10, 5, 3, 2],
            all_activations="relu",
            use_encoder_activations_in_alarm=True,
            use_latent_activations_in_encoder_activations=True,
            use_decoder_outputs_in_decoder_activations=True,
            encoder_dropout_frac=None,
            decoders_dropout_frac=None,
            alarm_dropout_frac=0.3,
            make_model_visualiations=False,
            autoencoder_l1=0,
            autoencoder_l2=0,
        )
        model.fit(
            x_train.drop(columns=["partition"]),
            epochs=None,
            autoencoder_epochs=4,
            alarm_epochs=4,
            batch_size=None,
            autoencoder_batch_size=1,
            alarm_batch_size=1,
            optimizer="adam",
            autoencoder_learning_rate=0.0001,
            alarm_learning_rate=0.0001,
            validation_split=1 / 5,
            n_noise_samples=None,
            noise_mean=10,
            plot_normal_vs_noise=True,
            log_with_wandb=False,
        )
        # model.save()

    anomalies = pd.DataFrame({"x1": [0, 1, 2, -1, 4, 100, -100, 8.22], "x2": [0, 1, 2, -1, 4, 100, -100, 2]})

    # predict the mixed data
    final_preds = np.round(model.predict(anomalies), 4)
    print(f"\nFinal anomaly probabilities:\n {final_preds}")
    model.predict_plot_anomalies(anomalies)
    plt.show()
