import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# from argue.models.argue import ARGUE
# from argue.utils.misc import set_seed

if __name__ == "__main__":
    # set_seed(1234)
    path = Path(os.environ["DATASETS"]) / "TEP" / "matlab_sim" / "processed"

    # make some data
    # fmt: off
    df_fault1 = pd.read_csv(path / "data_TEP_phase2_IDV1_run1.csv", index_col="Time")
    df_fault2 = pd.read_csv(path / "data_TEP_phase2_IDV2_run1.csv", index_col="Time")

    df = pd.read_csv(path / "data_TEP_phase1_run1.csv", index_col="Time")
    df["IDV"] = 0
    for IDV in range(1, 6):
        df_fault = pd.read_csv(path / f"data_TEP_phase2_IDV{IDV}_run1.csv", index_col="Time")
        df_fault[f"IDV"] = df_fault[f"IDV{IDV}"].mul(IDV).astype(int)
        df_fault = df_fault.drop(columns=[f"IDV{IDV}"])
        df = pd.concat([df, df_fault], axis=0)# .reset_index(drop=True)

    # df.index = pd.date_range(start="2000-01-01", periods=df.shape[0], freq="min")

    df_labels = df["IDV"]
    df = df.drop(columns=["IDV"])

    # split into train, validation, debugging and test sets
    X_train, X_test, y_train, y_test = train_test_split(df, df_labels, test_size=0.3, stratify=df_labels)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.33, stratify=y_test)
    X_test, X_debug, y_test, y_debug = train_test_split(X_test, y_test, test_size=0.5, stratify=y_test)


    # make PCA and scaling pipeline
    pipe = Pipeline([("scaler1", StandardScaler()),
                     ("PCA", PCA()),
                     ("scaler2", StandardScaler())
                     ])
    pipe.fit(X_train, y_train)
    X_train = pipe.transform(X_train)
    X_debug = pipe.transform(X_debug)
    X_val = pipe.transform(X_val)
    X_test = pipe.transform(X_test)
    # fmt: on

    # USE_SAVED_MODEL = True
    USE_SAVED_MODEL = False
    if USE_SAVED_MODEL:
        model = ARGUE().load()
    else:
        # call and fit model
        model = ARGUE(input_dim=2, number_of_decoders=2, latent_dim=1)
        model.build_model(
            encoder_hidden_layers=[6, 5, 4, 3, 2],
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
            make_model_visualiations=False,
        )
        model.fit(
            x_train.drop(columns=["partition"]),
            x_train["partition"],
            epochs=None,
            autoencoder_epochs=4,
            alarm_gating_epochs=4,
            batch_size=None,
            autoencoder_batch_size=1,
            alarm_gating_batch_size=1,
            optimizer="adam",
            ae_learning_rate=0.0001,
            alarm_gating_learning_rate=0.0001,
            autoencoder_decay_after_epochs=None,
            alarm_decay_after_epochs=None,
            gating_decay_after_epochs=None,
            decay_rate=0.5,
            fp_penalty=0,
            fn_penalty=0,
            validation_split=1 / 5,
            n_noise_samples=None,
            noise_mean=10,
            log_with_wandb=True,
        )
        # model.save()

    foo = model.ae_train_loss
    foo = model.alarm_val_loss

    anomalies = pd.DataFrame({"x1": [0, 1, 2, -1, 4, 100, -100, 8.22], "x2": [0, 1, 2, -1, 4, 100, -100, 2]})

    # predict the mixed data
    final_preds = np.round(model.predict(anomalies), 4)
    print("Alarm probabilities:\n ", np.round(model.predict_alarm_probabilities(anomalies), 4))
    print("\nGating weights:\n ", np.round(model.predict_gating_weights(anomalies), 3))
    print(f"\nFinal anomaly probabilities:\n {final_preds}")
    model.predict_plot_anomalies(anomalies, true_partitions=None, window_length=2)
    plt.show()
