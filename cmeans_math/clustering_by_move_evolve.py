import math

import cluster_centers
import entry_cluster_assignment
import covariances
import memberships
import accuracies
import density_functions
import clustering_results


MAX_LOSS_DELTA = 0.01
MAX_COUNT_TRAINING_STEPS = 10


def clustering_by_simple_mahalanobis_density(mat_entries, var_min_count_clusters, var_init_count_clusters):
    # generate random init cluster centers
    mat_cluster_centers = cluster_centers.random_cluster_centers(mat_entries=mat_entries,
                                                                 var_count_clusters=var_init_count_clusters)
    # assign entry to each cluster center
    mat_cluster_entry_indexes = entry_cluster_assignment.\
        manhattan_cluster_assignment(mat_entries=mat_entries, mat_cluster_centers=mat_cluster_centers)

    # get covariance matrices for each cluster
    ten_covariances = covariances.cluster_covariances(mat_entries, mat_cluster_entry_indexes, mat_cluster_centers)

    # get densities for the each cluster
    vec_clusters_densities = density_functions.simple_mahalanobis_density(mat_entries, mat_cluster_centers,
                                                                          ten_covariances, mat_cluster_entry_indexes)

    vec_total_densities = []
    vec_clusters_count = []

    var_old_loss = 0.0
    var_new_loss = sum(vec_clusters_densities) / var_init_count_clusters

    # loop for removing useless cluster centers
    while var_init_count_clusters > var_min_count_clusters:

        # do some clusters moving
        i = 0
        while math.fabs(var_old_loss - var_new_loss) > MAX_LOSS_DELTA and i <= MAX_COUNT_TRAINING_STEPS:
            i += 1
            mat_membership = memberships.mahalanobis_membership_matrix(mat_entries, mat_cluster_centers,
                                                                       ten_covariances)
            mat_membership = memberships.norm_membership_matrix_for_entries(mat_membership)

            cluster_centers.c_means_centers_moving(mat_cluster_centers, mat_entries, mat_membership,
                                                   mat_cluster_entry_indexes)

            mat_cluster_entry_indexes = entry_cluster_assignment. \
                mahalanobis_cluster_assignment(mat_entries, mat_cluster_centers, ten_covariances)

            ten_covariances = covariances.cluster_covariances(mat_entries, mat_cluster_entry_indexes,
                                                              mat_cluster_centers)

            vec_clusters_densities = density_functions.simple_mahalanobis_density(mat_entries, mat_cluster_centers,
                                                                                  ten_covariances,
                                                                                  mat_cluster_entry_indexes)
            var_old_loss = var_new_loss
            var_new_loss = sum(vec_clusters_densities) / var_init_count_clusters

        var_old_loss = 0.0

        print("\n\ncount clusters", var_init_count_clusters)
        print("total loss", sum(vec_clusters_densities) / var_init_count_clusters)
        vec_total_densities.append(var_new_loss)
        vec_clusters_count.append(var_init_count_clusters)

        var_max_loss_cluster_index = 0
        var_saved_len = 6
        for j in range(len(mat_cluster_entry_indexes)):
            if len(mat_cluster_entry_indexes[j]) < 5:
                if var_saved_len > len(mat_cluster_entry_indexes[j]):
                    var_saved_len = len(mat_cluster_entry_indexes[j])
                    var_max_loss_cluster_index = j

        if var_saved_len < 6:
            del mat_cluster_centers[var_max_loss_cluster_index]
            del ten_covariances[var_max_loss_cluster_index]
        else:
            var_max_loss_cluster_index = vec_clusters_densities.index(min(vec_clusters_densities))
            del mat_cluster_centers[var_max_loss_cluster_index]
            del ten_covariances[var_max_loss_cluster_index]

        var_init_count_clusters -= 1

        print("cluster losses:", vec_clusters_densities)
        vec_count_entries_in_clusters = []
        for x in mat_cluster_entry_indexes:
            vec_count_entries_in_clusters.append(len(x))
        print("count elems:", vec_count_entries_in_clusters)
        print("deleted cluster:", var_max_loss_cluster_index)

        mat_cluster_entry_indexes = entry_cluster_assignment. \
            mahalanobis_cluster_assignment(mat_entries, mat_cluster_centers, ten_covariances)
        ten_covariances = covariances.cluster_covariances(mat_entries, mat_cluster_entry_indexes, mat_cluster_centers)

    print("\n\ncount clusters", var_init_count_clusters)
    print("total loss", sum(vec_clusters_densities) / var_init_count_clusters)
    vec_total_densities.append(var_new_loss)
    vec_clusters_count.append(var_init_count_clusters)

    return clustering_results.ClusteringResults(vec_clusters_count=vec_clusters_count,
                                                vec_total_losses=vec_total_densities,
                                                mat_cluster_centers=mat_cluster_centers,
                                                mat_cluster_entry_indexes=mat_cluster_entry_indexes)
