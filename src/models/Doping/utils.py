import matplotlib.pyplot as plt
import numpy as np
from numpy import random
from pandas import DataFrame
from scipy import spatial
from scipy import stats


def _get_row_from_df_as_array(df, row):
    return np.array(df.iloc[row, :]).reshape((1, df.shape[1]))


def upsample_points_using_random_noise(df_original_points_to_upsample: DataFrame,
                                       N_new_samples_to_add_per_original_point: int = 1,
                                       std_dev_noise_points: float = 0.1):
    sampled_noise_points = []
    N_original_rows, N_features = df_original_points_to_upsample.shape
    for i in range(N_original_rows):
        data_point_i = _get_row_from_df_as_array(df_original_points_to_upsample, i)
        new_noise_point = random.normal(loc=data_point_i,
                                        scale=std_dev_noise_points,
                                        size=(N_new_samples_to_add_per_original_point, N_features))
        sampled_noise_points.append(new_noise_point)
    sampled_noise_points = np.array(sampled_noise_points).reshape((-1, N_features))
    return sampled_noise_points


def upsample_points_using_nearest_neighbors(df_original_points_to_upsample: DataFrame,
                                            N_new_samples_to_add_per_original_point: int = 1,
                                            number_of_neighbours_to_use_per_point: int = 3,
                                            add_noise_to_new_samples: bool = True,
                                            noise_magnitude: float = 1.0):
    def _find_dists_and_indices_of_nearest_k_neighbours():
        neighbour_tree_model = spatial.cKDTree(df_original_points_without_row_i)
        return neighbour_tree_model.query(data_point_i, k=number_of_neighbours_to_use_per_point)

    def _choose_neighbour_randomly():
        neighbour = random.random_integers(0, number_of_neighbours_to_use_per_point - 1)
        return neighbour_indices[:, neighbour], neighbour_distances[:, neighbour]

    sampled_points = []
    df_original_points_copy = df_original_points_to_upsample.copy().reset_index(drop=True)
    N_original_rows, N_features = df_original_points_copy.shape
    for i in range(N_original_rows):
        df_original_points_without_row_i = df_original_points_copy.drop(i).reset_index(drop=True)
        data_point_i = _get_row_from_df_as_array(df_original_points_copy, row=i)
        neighbour_distances, neighbour_indices = _find_dists_and_indices_of_nearest_k_neighbours()
        for k in range(N_new_samples_to_add_per_original_point):
            # select one of the k nearest neighbours with probability 1/k
            if number_of_neighbours_to_use_per_point > 1:
                chosen_neighbour_index, chosen_neighbour_distance = _choose_neighbour_randomly()
            else:
                chosen_neighbour_distance = neighbour_distances
                chosen_neighbour_index = neighbour_indices
            chosen_neighbour = _get_row_from_df_as_array(df_original_points_without_row_i, row=chosen_neighbour_index)
            random_percentage_distance = random.uniform(0, 1, (1, 1))
            # TODO wrap below line in a function
            new_sample_between_neighbours = random_percentage_distance * (chosen_neighbour - data_point_i) + data_point_i
            if add_noise_to_new_samples:
                # TODO wrap in function
                 new_sample_between_neighbours = new_sample_between_neighbours + np.random.normal(0, noise_magnitude * chosen_neighbour_distance, (1, N_features))
            sampled_points.append(new_sample_between_neighbours)
    sampled_points = np.array(sampled_points).reshape((-1, N_features))
    return sampled_points

    # z_new_samples = []
    # z_edge_copy = df_original_points_to_upsample.copy().reset_index(drop=True)
    # for i in range(df_original_points_to_upsample.shape[0]):
    #     z_edge_without_z_i = z_edge_copy.drop(i).reset_index(drop=True)
    #     z_i = np.array(z_edge_copy.iloc[i, :]).reshape((1, 2))
    #     dists, idx_nns = spatial.cKDTree(z_edge_without_z_i).query(z_i, k=number_of_neighbours_to_use_per_point)
    #     for k in range(N_new_samples_to_add_per_original_point):
    #         # select one of the k nearest neighbours with probability 1/k
    #         if number_of_neighbours_to_use_per_point > 1:
    #             u = random.random_integers(0, number_of_neighbours_to_use_per_point - 1)
    #             idx_nn = idx_nns[:, u]
    #             dist = dists[:, u]
    #         else:
    #             dist = dists
    #             idx_nn = idx_nns
    #         z_nn = np.array(z_edge_without_z_i.iloc[idx_nn, :])
    #         alpha = random.uniform(0, 1, (1, 1))
    #         z_new = alpha * (z_nn - z_i) + z_i
    #         if add_noise_to_new_samples:
    #              z_new = z_new + np.random.normal(0, noise_magnitude * dist, (1, 2))
    #         z_new_samples.append(z_new)
    # z_new_samples = np.array(z_new_samples).reshape((-1, 2))
    # return z_new_samples


#
# def upsample_edge_nn_noisy(df_original_points_to_upsample: DataFrame, N_new_samples_to_add_per_original_point: int = 1):
#     z_new_samples = []
#     for i in range(df_original_points_to_upsample.shape[0]):
#         z_edge_without_z_i = df_original_points_to_upsample.drop(i).reset_index(drop=True)
#         z_i = np.array(df_original_points_to_upsample.iloc[i, :]).reshape((1, 2))
#         dist, idx_nn = spatial.cKDTree(z_edge_without_z_i).query(z_i)
#         z_nn = np.array(z_edge_without_z_i.iloc[idx_nn, :])
#         for k in range(N_new_samples_to_add_per_original_point):
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
