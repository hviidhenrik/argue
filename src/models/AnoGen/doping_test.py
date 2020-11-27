import pandas as pd
from src.models.AnoGen.doping_utils import *

# sample gaussian mixture dist
# z_train = pd.DataFrame(random.normal(0, 1, (5000, 2)), columns=["z0", "z1"])
x1 = random.multivariate_normal([0, 0], [[0.1, -1.5], [-1.5, 10]], 2500)
x2 = random.multivariate_normal([0, 0], [[0.1, 1.5], [1.5, 10]], 2500)
z_train = pd.DataFrame(np.vstack([x1, x2]), columns=["z0", "z1"])
# plt.scatter(z_train.iloc[:, 0], z_train.iloc[:, 1])
# plt.show()

z_edge = find_edge_region_kde(z_train)
# plt.scatter(z_edge.iloc[:, 0], (z_edge.iloc[:, 1]))
# plt.show()

# sample independent gaussian
x1 = random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 5000)
z_train = pd.DataFrame(x1, columns=["z0", "z1"])
# plt.scatter(z_train.iloc[:, 0], z_train.iloc[:, 1])
# plt.show()

z_edge = find_edge_region_kde(z_train, 0.03)
# plt.scatter(z_edge.iloc[:, 0], (z_edge.iloc[:, 1]))
# plt.show()

# sample complicated distribution
x1 = random.multivariate_normal([0, 0], [[0.1, -1.5], [-1.5, 10]], 2500)
x2 = np.vstack([random.exponential(1, 2500), random.exponential(1, 2500)]).reshape((-1, 2))
z_train = pd.DataFrame(np.vstack([x1, x2]), columns=["z0", "z1"])
# plt.scatter(z_train.iloc[:, 0], z_train.iloc[:, 1])
# plt.show()

z_edge = find_edge_region_kde(z_train, 0.1)
plt.scatter(z_edge.iloc[:, 0], (z_edge.iloc[:, 1]))
plt.show()

## deliberately oversample this edge region

# use nearest neighbour interpolation to oversample
z_new_samples = upsample_edge_nn(z_edge, N_samples_per_point=3, k_neighbors=1, add_noise=False, noise_magnitude=0.4)
plt.scatter(z_edge.iloc[:, 0], (z_edge.iloc[:, 1]), label="normal")
plt.scatter(z_new_samples[:, 0], z_new_samples[:, 1], s=10, alpha=0.7, c="black", label="samples")
plt.legend()
plt.show()

z_new_samples = upsample_edge_nn(z_edge, N_samples_per_point=3, k_neighbors=3, add_noise=True, noise_magnitude=0.6)
plt.scatter(z_edge.iloc[:, 0], (z_edge.iloc[:, 1]), label="normal")
plt.scatter(z_new_samples[:, 0], z_new_samples[:, 1], s=10, alpha=0.7, c="black", label="samples")
plt.legend()
plt.show()
