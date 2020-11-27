import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from scipy import spatial
from scipy.spatial.distance import cdist
import pandas as pd


# sample normal dist
z_train = pd.DataFrame(random.normal(0, 1, (5000, 2)), columns=["z0", "z1"])
z_train = pd.DataFrame(random.multivariate_normal([0,0], [[1,0.9], [0.9,1]], 5000), columns=["z0", "z1"])
plt.scatter(z_train.iloc[:,0], z_train.iloc[:, 1])
plt.show()

# calculate norm for all of them
z_train["norm"] = np.linalg.norm(z_train, ord=2, axis=1).reshape((-1, 1))

# filter away those with alpha < norm(z) < beta
beta = np.mean(z_train["norm"]) + 3 * np.std(z_train["norm"])  # np.quantile(z_edge["norm"], 0.7)
z_edge = z_train[(z_train["norm"] < beta)]
alpha = np.quantile(z_edge["norm"], 0.9)
z_edge = z_edge[(z_edge["norm"] > alpha)]
plt.scatter(z_edge.iloc[:, 0], (z_edge.iloc[:, 1]))
plt.show()

## deliberately oversample this edge region
z_edge = z_edge.drop("norm", axis=1).reset_index(drop=True)
def upsample_edge_gaussian(z_edge, K: int = 1, stddev: float = 0.1):
    z_new_samples = []
    for i in range(z_edge.shape[0]):
        z_i = np.array(z_edge.iloc[i, :]).reshape((1, 2))
        u = random.normal(z_i, stddev, (K, 2))
        z_new_samples.append(u)
    z_new_samples = np.array(z_new_samples).reshape((-1, 2))
    return z_new_samples

z_new_samples = upsample_edge_gaussian(z_edge)
print(z_new_samples.shape[0])
plt.scatter(z_edge.iloc[:, 0], (z_edge.iloc[:, 1]), label="normal")
plt.scatter(z_new_samples[:, 0], z_new_samples[:, 1], c="black", marker="^", label="sampled")
plt.show()


def upsample_edge_nn(z_edge, K: int = 1):
    z_new_samples = []
    for i in range(z_edge.shape[0]):
        z_edge_without_z_i = z_edge.drop(i).reset_index(drop=True)
        z_i = np.array(z_edge.iloc[i, :]).reshape((1, 2))
        _, idx_nn = spatial.cKDTree(z_edge_without_z_i).query(z_i)
        z_nn = np.array(z_edge_without_z_i.iloc[idx_nn,:])
        for k in range(K):
            alpha = random.uniform(0,1, (1,1))
            z_new = alpha*(z_nn - z_i) + z_i
            z_new_samples.append(z_new)
    z_new_samples = np.array(z_new_samples).reshape((-1, 2))
    return z_new_samples

z_new_samples = upsample_edge_nn(z_edge, 10)
print(z_new_samples.shape[0])
plt.scatter(z_edge.iloc[:, 0], (z_edge.iloc[:, 1]), label="normal")
plt.scatter(z_new_samples[:, 0], z_new_samples[:, 1], c="red", marker="^", label="sampled")
plt.show()


def upsample_edge_nn_noisy(z_edge, K: int = 1):
    z_new_samples = []
    for i in range(z_edge.shape[0]):
        z_edge_without_z_i = z_edge.drop(i).reset_index(drop=True)
        z_i = np.array(z_edge.iloc[i, :]).reshape((1, 2))
        dist, idx_nn = spatial.cKDTree(z_edge_without_z_i).query(z_i)
        z_nn = np.array(z_edge_without_z_i.iloc[idx_nn,:])
        for k in range(K):
            alpha = random.uniform(0,1, (1,1))
            z_new = alpha*(z_nn - z_i) + z_i + np.random.normal(0, dist, (1,2))
            z_new_samples.append(z_new)
    z_new_samples = np.array(z_new_samples).reshape((-1, 2))
    return z_new_samples

z_new_samples = upsample_edge_nn_noisy(z_edge, 1)
print(z_new_samples.shape[0])
plt.scatter(z_edge.iloc[:, 0], (z_edge.iloc[:, 1]), label="normal")
plt.scatter(z_new_samples[:, 0], z_new_samples[:, 1], c="red", marker="^", label="sampled")
plt.show()
