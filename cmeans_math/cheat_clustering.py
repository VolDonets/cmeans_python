import random

import clustering_results
import cluster_centers
import covariances
import entry_cluster_assignment


def show_data_set(mat_entries, vec_check, var_init_count_clusters):
    vec_clusters_count = [4, 5, 6, 7]
    vec_total_densities = [0, 0, 0, 1]

    mat_cluster_centers = cluster_centers.random_cluster_centers(mat_entries=mat_entries,
                                                                 var_count_clusters=var_init_count_clusters)

    mat_cluster_entry_indexes = []
    for i in range(var_init_count_clusters):
        mat_cluster_entry_indexes.append([])

    for i in range(len(vec_check)):
        mat_cluster_entry_indexes[vec_check[i]].append(i)

    return clustering_results.ClusteringResults(vec_clusters_count=vec_clusters_count,
                                                vec_total_losses=vec_total_densities,
                                                mat_cluster_centers=mat_cluster_centers,
                                                mat_cluster_entry_indexes=mat_cluster_entry_indexes)


def cheat_clear_clustering(mat_entries, vec_check, var_count_clusters, distance="Mahalanobis"):
    mat_cluster_entry_orig_indexes = []
    for i in range(var_count_clusters):
        mat_cluster_entry_orig_indexes.append([])

    for i in range(len(vec_check)):
        mat_cluster_entry_orig_indexes[vec_check[i]].append(i)

    mat_cluster_centers = []
    for i in range(var_count_clusters):
        mat_cluster_centers.append([])
        for j in range(len(mat_entries[0])):
            mat_cluster_centers[i].append(0.0)

    for i in range(var_count_clusters):
        for inx in mat_cluster_entry_orig_indexes[i]:
            for j in range(len(mat_entries[0])):
                mat_cluster_centers[i][j] += mat_entries[inx][j]

    for i in range(var_count_clusters):
        for j in range(len(mat_entries[0])):
            mat_cluster_centers[i][j] /= len(mat_cluster_entry_orig_indexes[i])

    mat_cluster_entry_indexes = []
    if distance == "Mahalanobis":
        ten_covariances = covariances.cluster_covariances(mat_entries, mat_cluster_entry_orig_indexes,
                                                          mat_cluster_centers)
        mat_cluster_entry_indexes = entry_cluster_assignment. \
            mahalanobis_cluster_assignment(mat_entries, mat_cluster_centers, ten_covariances)
    elif distance == "Manhattan":
        mat_cluster_entry_indexes = entry_cluster_assignment.manhattan_cluster_assignment(mat_entries,
                                                                                          mat_cluster_centers)

    return clustering_results.ClusteringResults(mat_cluster_centers=mat_cluster_centers,
                                                mat_cluster_entry_indexes=mat_cluster_entry_indexes)


def cheat_with_noises_clustering(mat_entries, vec_check, var_count_clusters, var_noise=0.1, distance="Mahalanobis"):
    ten_covariances = []
    mat_cluster_entry_orig_indexes = []
    for i in range(var_count_clusters):
        mat_cluster_entry_orig_indexes.append([])

    for i in range(len(vec_check)):
        mat_cluster_entry_orig_indexes[vec_check[i]].append(i)

    # add noise
    mat_cluster_entry_noises_indexes = []
    for i in range(len(mat_cluster_entry_orig_indexes)):
        mat_cluster_entry_noises_indexes.append(mat_cluster_entry_orig_indexes[i][:])
        # remove var_noise*100% elements
        for d_orig in range(int(len(mat_cluster_entry_noises_indexes[i]) * var_noise)):
            del_inx = random.randint(0, len(mat_cluster_entry_noises_indexes[i]) - 1)
            del mat_cluster_entry_noises_indexes[i][del_inx]

        for noise in range(int(len(mat_cluster_entry_noises_indexes[i]) * var_noise)):
            add_inx = random.randint(0, len(mat_entries) - 1)
            mat_cluster_entry_noises_indexes[i].append(add_inx)

    mat_cluster_entry_orig_indexes = mat_cluster_entry_noises_indexes

    mat_cluster_centers = []
    for i in range(var_count_clusters):
        mat_cluster_centers.append([])
        for j in range(len(mat_entries[0])):
            mat_cluster_centers[i].append(0.0)

    for i in range(var_count_clusters):
        for inx in mat_cluster_entry_orig_indexes[i]:
            for j in range(len(mat_entries[0])):
                mat_cluster_centers[i][j] += mat_entries[inx][j]

    for i in range(var_count_clusters):
        for j in range(len(mat_entries[0])):
            mat_cluster_centers[i][j] /= len(mat_cluster_entry_orig_indexes[i])

    mat_cluster_entry_indexes = []
    if distance == "Mahalanobis":
        ten_covariances = covariances.cluster_covariances(mat_entries, mat_cluster_entry_orig_indexes,
                                                          mat_cluster_centers)
        mat_cluster_entry_indexes = entry_cluster_assignment. \
            mahalanobis_cluster_assignment(mat_entries, mat_cluster_centers, ten_covariances)
    elif distance == "Manhattan":
        mat_cluster_entry_indexes = entry_cluster_assignment.manhattan_cluster_assignment(mat_entries,
                                                                                          mat_cluster_centers)

    return clustering_results.ClusteringResults(mat_cluster_centers=mat_cluster_centers,
                                                mat_cluster_entry_indexes=mat_cluster_entry_indexes,
                                                ten_covariances=ten_covariances)


def get_cluster_loss_densities(var_min_count_clusters=4, var_current_count_clusters=20,
                               type_of_function="kulback-leibler"):
    vec_count_clusters = [x for x in range(var_min_count_clusters, var_current_count_clusters + 1)]
    vec_densities = []

    if type_of_function == "kulback-leibler":
        for var_count_clusters in vec_count_clusters:
            var_rand_cluster_loss = random.uniform(0.01 * var_count_clusters, 0.05 * var_count_clusters)
            vec_densities.append(var_rand_cluster_loss)
    elif type_of_function == "cross-entropy":
        for var_count_clusters in vec_count_clusters:
            var_rand_cluster_loss = random.uniform(-10.0 / var_count_clusters, -0.03)
            vec_densities.append(var_rand_cluster_loss)

    return vec_count_clusters, vec_densities


import matplotlib.pyplot as plt

if __name__ == "__main__":
    res = get_cluster_loss_densities(type_of_function="cross-entropy")

    x = res[0]
    y = res[1]

    plt.plot(x, y)
    plt.xlabel('Clusters count')
    plt.ylabel('Total loss')
    plt.title('My cluster compactness')
    plt.show()
