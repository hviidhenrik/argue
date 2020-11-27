import matplotlib.pyplot as plt
import numpy as np
from numpy import random
from pandas import DataFrame
from scipy import spatial
from scipy import stats


def upsample_edge_gaussian(z_edge: DataFrame, K: int = 1, stddev: float = 0.1):
    z_new_samples = []
    for i in range(z_edge.shape[0]):
        z_i = np.array(z_edge.iloc[i, :]).reshape((1, 2))
        u = random.normal(z_i, stddev, (K, 2))
        z_new_samples.append(u)
    z_new_samples = np.array(z_new_samples).reshape((-1, 2))
    return z_new_samples


def upsample_edge_nn(z_edge: DataFrame,
                     K: int = 1,
                     add_noise: bool = True,
                     noise_magnitude: float = 1.0):
    z_new_samples = []
    z_edge_copy = z_edge.copy().reset_index(drop=True)
    for i in range(z_edge.shape[0]):
        z_edge_without_z_i = z_edge_copy.drop(i).reset_index(drop=True)
        z_i = np.array(z_edge_copy.iloc[i, :]).reshape((1, 2))
        dist, idx_nn = spatial.cKDTree(z_edge_without_z_i).query(z_i)
        z_nn = np.array(z_edge_without_z_i.iloc[idx_nn, :])
        for k in range(K):
            alpha = random.uniform(0, 1, (1, 1))
            z_new = alpha * (z_nn - z_i) + z_i
            if add_noise:
                 z_new = z_new + np.random.normal(0, noise_magnitude * dist, (1, 2))
            z_new_samples.append(z_new)
    z_new_samples = np.array(z_new_samples).reshape((-1, 2))
    return z_new_samples

#
# def upsample_edge_nn_noisy(z_edge: DataFrame, K: int = 1):
#     z_new_samples = []
#     for i in range(z_edge.shape[0]):
#         z_edge_without_z_i = z_edge.drop(i).reset_index(drop=True)
#         z_i = np.array(z_edge.iloc[i, :]).reshape((1, 2))
#         dist, idx_nn = spatial.cKDTree(z_edge_without_z_i).query(z_i)
#         z_nn = np.array(z_edge_without_z_i.iloc[idx_nn, :])
#         for k in range(K):
#             alpha = random.uniform(0, 1, (1, 1))
#             z_new = alpha * (z_nn - z_i) + z_i + np.random.normal(0, dist / 2, (1, 2))
#             z_new_samples.append(z_new)
#     z_new_samples = np.array(z_new_samples).reshape((-1, 2))
#     return z_new_samples

def plot_kde(z_train):
    z0_min, z0_max = z_train["z0"].min() , z_train["z0"].max()
    z1_min, z1_max = z_train["z1"].min() , z_train["z1"].max()
    X, Y = np.mgrid[z0_min:z0_max:100j, z1_min:z1_max:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([z_train["z0"], z_train["z1"]])
    kernel = stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)
    fig, ax = plt.subplots()
    ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
              extent=[z0_min, z0_max, z1_min, z1_max])
    ax.plot(z_train["z0"], z_train["z1"], 'k.', markersize=2)
    ax.set_xlim([z0_min, z0_max])
    ax.set_ylim([z1_min, z1_max])
    plt.show()


def find_edge_region_kde(z_train, quantile: float = 0.1):
    z_edge = z_train.copy()
    points = np.vstack([z_edge["z0"], z_edge["z1"]])
    kernel = stats.gaussian_kde(points)
    z_edge["pdf"] = kernel.evaluate(z_edge.T)
    z_edge = z_edge[z_edge["pdf"] < np.quantile(z_edge["pdf"], quantile)]
    return z_edge.drop("pdf", axis=1)
