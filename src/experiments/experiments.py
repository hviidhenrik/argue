"""
Anomaly Generation Framework - AnoGen

This script explores the possibilities of using a variational autoencoder (VAE)
to learn the conditional distribution of high dimensional data. This enables
sampling of points from the latent space, which are unlikely to observe according
to the learned multivariate distribution.

VAE adapted from: http://physics.bu.edu/~pankajm/ML-Notebooks/HTML/NB19_CXVII-Keras_VAE_MNIST.html

Also, for more in-depth on VAE code have a look at:
https://tiao.io/post/tutorial-on-variational-autoencoders-with-a-concise-keras-implementation/#fnref:12

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pdm.utils.definitions import time_interval
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src.models.AnoGen.utility_functions import (fit_VAE,
                                                           fit_AE,
                                                           append_anomalies,
                                                           predict_AE_anomalies,
                                                           filter_latent_samples)

# tell pandas to display all columns when printing dataframes
pd.set_option('display.max_columns', None)

"""
Import data
"""
df = pd.read_csv("train-data-large.csv", index_col="timelocal")
df = df.dropna()
col_names = df.columns.values

"""
Prepare data for training
"""

# split data
x_train, x_test = train_test_split(df, test_size=0.2, shuffle=False)
x_val, x_test = train_test_split(x_test, test_size=0.5, shuffle=False)

# scale data
scaler_train = MinMaxScaler()
x_train_scaled = scaler_train.fit_transform(x_train)
x_val_scaled = scaler_train.transform(x_val)
x_test_scaled = scaler_train.transform(x_test)

"""
Train the VAE
"""
print("\n--- Training Variational Autoencoder for generating anomalies...")

vae_latent_dim = 2
encoder, decoder, vae = fit_VAE(x_train_scaled,
                                x_val_scaled,
                                intermediate_dim=12,
                                latent_dim=vae_latent_dim,
                                batch_size=128,
                                epochs=2,
                                early_stopping=True,
                                kl_warmup=30,
                                latent_stddev=0.008,
                                plot_history=False,
                                activation="elu")

"""
Train ordinary Autoencoder for MSE based anomaly detection
"""
print("\n--- Training ordinary Autoencoder for anomaly detection...")
ae_latent_dim = 2
ae, mse_train, mse_val, ae_latent_space = fit_AE(x_train_scaled,
                                                 x_val_scaled,
                                                 intermediate_dim=12,
                                                 latent_dim=ae_latent_dim,
                                                 batch_size=128,
                                                 epochs=2,
                                                 early_stopping=True,
                                                 plot_history=False,
                                                 plot_latent=False,
                                                 activation="elu")

"""
Sample anomaly points from the VAE latent space based on the training data
"""
vae_latent_space = encoder.predict(x_train_scaled)
N_samples = 5000
df_z_samples = pd.DataFrame({"z0": np.random.uniform(-1, 1, N_samples),
                             "z1": np.random.uniform(-1, 1, N_samples)})
decoded_latent_samples = decoder.predict(df_z_samples)
decoded_latent_samples = scaler_train.inverse_transform(decoded_latent_samples)
df_z_samples_decoded = pd.DataFrame(decoded_latent_samples, columns=df.columns)

# filter unrealistic anomalies away based on domain knowledge and observed ranges in training set
domain_filter = {"kv_flow": [0, 18000],
                 "flush_indicator": [0, 1]}
df_z_and_preds = pd.concat([df_z_samples, df_z_samples_decoded], axis=1)
df_z_and_preds = filter_latent_samples(df_z_and_preds, df, domain_filter)
N_samples = df_z_and_preds.shape[0]

"""
Plot the learned latent space with sampled anomaly points, color coding by original features
"""

for coloring_col in range(1):
    plt.scatter(vae_latent_space[:, 0], vae_latent_space[:, 1],
                c=x_train.iloc[:, coloring_col], cmap='jet', s=10)
    plt.xlabel("z_0")
    plt.ylabel("z_1")
    clb = plt.colorbar()
    clb.set_label(col_names[coloring_col], rotation=0, labelpad=-30, y=1.05)
    plt.scatter(df_z_and_preds["z0"], df_z_and_preds["z1"],
                s=20, color="black", marker="^", label="Sampled anomaly point")
    plt.title("VAE latent space (training data)")
    plt.legend()
    # plt.savefig('VAE_latent_[{}].png'.format(feature_cols[coloring_col]))
    plt.show()

"""
Make dataframe with nominal datapoints and generated anomalies
"""

# prepare prediction data by pasting sampled anomalies to it, scaling and removing synthetic anomaly column
df_ready_to_predict = append_anomalies(x_test, df_z_and_preds.iloc[:, vae_latent_dim:])
df_synth_anomaly = df_ready_to_predict[["synthetic_anomaly"]]

df_ready_to_predict = df_ready_to_predict.drop("synthetic_anomaly", axis=1)
df_ready_to_predict_scaled = scaler_train.transform(df_ready_to_predict)

# predict the new data using the ordinary Autoencoder for anomaly detection and put back synthetic anomaly column
anomaly_threshold = np.quantile(mse_train, 0.98)
preds, df_anomalies = predict_AE_anomalies(ae, df_ready_to_predict_scaled, anomaly_threshold)
df_preds = pd.DataFrame(scaler_train.inverse_transform(preds), columns=col_names)
df_synth_anomaly.index = df_preds.index
df_preds = pd.concat([df_preds, df_anomalies, df_synth_anomaly], axis=1)
print(df_preds.sort_values("AE_mse", ascending=False).head(10))
print(df_preds.sort_values("AE_mse", ascending=False).describe())

df_anomalies.index = df_synth_anomaly.index
df_anomalies = pd.concat([df_anomalies, df_synth_anomaly], axis=1)

"""
Plot as timeseries the dataframe with both nominal points and generated anomalies
"""

# plot the predicted data to see which anomalies are caught by the model
axes = df_anomalies.iloc[-N_samples - 100:, ].plot(subplots=True, layout=(3, 1))
plt.suptitle("SSV CWP 10\nSynthetic anomalies starting after black vertical line")
for c in axes:
    for ax in c:
        ax.axvline(x=df_anomalies.index[-N_samples - 100], color='black', linestyle='--')
        ax.legend(loc="upper left")
plt.show()

""" 
Plot the latent space again, but this time with the points flagged as anomalies colored by anomaly indicator and MSE
"""

df_anomaly_count = df_anomalies.iloc[-N_samples:, ]
df_anomaly_count.index = df_z_and_preds.index
df_anomalies_flagged = pd.concat([df_z_and_preds[["z0", "z1"]], df_anomaly_count], axis=1)

# color code by anomaly indicator (binary 0,1) and set size by MSE
plt.scatter(vae_latent_space[:, 0], vae_latent_space[:, 1], c="black", s=10,
            label="VAE training\nlatent space", alpha=0.5)
mse_color_scale = MinMaxScaler((2, 20)).fit_transform(df_anomalies_flagged[["AE_mse"]])
plt.scatter(df_anomalies_flagged["z0"], df_anomalies_flagged["z1"],
            s=mse_color_scale, c=df_anomalies_flagged["AE_anomaly"],
            cmap='jet', marker="^", label="Sampled anomaly\n(from VAE latent)")
plt.xlabel("z_0")
plt.ylabel("z_1")
clb = plt.colorbar()
clb.set_label("AE_anomaly", rotation=0, labelpad=-20, y=1.1)
plt.title("VAE latent space (training data)\nwith latent samples colored by anomaly indicator")
plt.legend(ncol=2)
# plt.savefig('AD_latent_[{}].png'.format(feature_cols[coloring_col]))
plt.show()

# color code by MSE
mse_color_scale = MinMaxScaler((0, 100)).fit_transform(df_anomalies_flagged[["AE_mse"]])
plt.scatter(df_anomalies_flagged["z0"], df_anomalies_flagged["z1"],
            cmap='jet', marker="s", label="Sampled anomaly\n(from VAE latent)",
            c=mse_color_scale, s=10)
clb = plt.colorbar()
plt.scatter(vae_latent_space[:, 0], vae_latent_space[:, 1], c="black", s=10,
            label="VAE training\nlatent space", alpha=0.5)
plt.xlabel("z_0")
plt.ylabel("z_1")
clb.set_label("Autoencoder\nMSE", rotation=0, labelpad=-20, y=1.15)
plt.title("VAE latent space (training data)\nwith latent samples colored by MSE")
plt.legend(ncol=2)
# plt.savefig('AD_latent_[{}].png'.format(feature_cols[coloring_col]))
plt.show()

# color code by features
# df_latent_and_reconstructions = pd.concat([df_anomalies_latent, df_z_samples_decoded], axis=1)
col_names = list(df_z_and_preds.drop(["z0", "z1"], axis=1).columns.values)[0:1]
for coloring_col in col_names:
    plt.scatter(df_z_and_preds["z0"], df_z_and_preds["z1"],
                cmap='bwr', marker="s", label="Sampled anomaly\n(from VAE latent)",
                c=df_z_and_preds[coloring_col], s=10)
    clb = plt.colorbar()
    clb.set_label(coloring_col, rotation=0, labelpad=-20, y=1.1)
    plt.scatter(vae_latent_space[:, 0], vae_latent_space[:, 1], c="black", s=10,
                label="VAE training\nlatent space", alpha=0.5)
    plt.xlabel("z_0")
    plt.ylabel("z_1")
    plt.title("VAE latent space (training data)\nwith latent samples colored by feature")
    plt.legend(ncol=2)
    # plt.savefig('AD_latent_[{}].png'.format(feature_cols[coloring_col]))
    plt.show()

print("\n... End of VAE script")
