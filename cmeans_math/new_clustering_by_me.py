import math

import cluster_centers
import entry_cluster_assignment
import covariances
import memberships
import accuracies
import density_functions
import clustering_results
import loss_functions
import divergences


MAX_LOSS_DELTA = 0.01
MAX_COUNT_TRAINING_STEPS = 10


def loss_fix(var_x):
    var_y = -0.0004 * (var_x ** 3) + 0.0106 * (var_x ** 2) - 0.0080 * var_x - 0.1136
    # var_y = 1.2986 - 6.3906 / var_x
    return var_y


def clustering_by_density(mat_entries, var_min_count_clusters, var_init_count_clusters,
                          distance="Manhattan"):
    # generate random init cluster centers
    mat_cluster_centers = cluster_centers.random_cluster_centers(mat_entries=mat_entries,
                                                                 var_count_clusters=var_init_count_clusters)
    # assign entry to each cluster center
    mat_cluster_entry_indexes = entry_cluster_assignment. \
        manhattan_cluster_assignment(mat_entries=mat_entries, mat_cluster_centers=mat_cluster_centers)
    mat_cluster_entry_indexes = entry_cluster_assignment.balance_cluster_assignment(mat_cluster_entry_indexes)

    # get covariance matrices for each cluster
    if "Mahalanobis" in distance:
        ten_covariances = covariances.cluster_covariances(mat_entries, mat_cluster_entry_indexes, mat_cluster_centers)

    vec_clusters_densities = []
    mat_membership = []

    if distance == "Manhattan" or distance == "InverseManhattan":
        mat_membership = memberships.manhattan_membership_matrix(mat_entries, mat_cluster_centers)
        mat_membership = memberships.norm_membership_matrix_for_entries(mat_membership)

    if distance == "SimpleManhattan":
        vec_clusters_densities = density_functions.\
            simple_manhattan_density(mat_entries, mat_cluster_centers,
                                     mat_cluster_entry_indexes)
    elif distance == "Manhattan":
        vec_clusters_densities = density_functions.\
            manhattan_density(mat_membership, mat_entries, mat_cluster_centers,
                              mat_cluster_entry_indexes)
    elif distance == "InverseManhattan":
        vec_clusters_densities = density_functions.\
            inverse_manhattan_density(mat_membership, mat_entries,
                                      mat_cluster_centers, mat_cluster_entry_indexes)

    vec_total_densities = []
    vec_clusters_count = []

    var_old_loss = 0.0
    # var_new_loss = sum(vec_clusters_densities) / var_init_count_clusters
    var_new_loss = (sum(vec_clusters_densities) / var_init_count_clusters) * loss_fix(var_init_count_clusters)
    # var_new_loss = sum(vec_clusters_densities) * loss_fix(var_init_count_clusters)

    # loop for removing useless cluster centers
    while var_init_count_clusters > var_min_count_clusters:

        # do some clusters moving
        i = 0
        while math.fabs(var_old_loss - var_new_loss) > MAX_LOSS_DELTA and i <= MAX_COUNT_TRAINING_STEPS:
            i += 1
            if "Manhattan" in distance:
                mat_membership = memberships.\
                    manhattan_membership_matrix(mat_entries, mat_cluster_centers)
            mat_membership = memberships.norm_membership_matrix_for_entries(mat_membership)
            cluster_centers.c_means_centers_moving(mat_cluster_centers,
                                                   mat_entries, mat_membership,
                                                   mat_cluster_entry_indexes)

            if "Manhattan" in distance:
                mat_cluster_entry_indexes = entry_cluster_assignment. \
                    manhattan_cluster_assignment(mat_entries, mat_cluster_centers)
            mat_cluster_entry_indexes = entry_cluster_assignment.balance_cluster_assignment(mat_cluster_entry_indexes)

            if distance == "Manhattan" or distance == "InverseManhattan":
                mat_membership = memberships.manhattan_membership_matrix(mat_entries, mat_cluster_centers)
                mat_membership = memberships.norm_membership_matrix_for_entries(mat_membership)

            if distance == "SimpleManhattan":
                vec_clusters_densities = density_functions. \
                    simple_manhattan_density(mat_entries, mat_cluster_centers,
                                             mat_cluster_entry_indexes)
            elif distance == "Manhattan":
                vec_clusters_densities = density_functions. \
                    manhattan_density(mat_membership, mat_entries, mat_cluster_centers,
                                      mat_cluster_entry_indexes)
            elif distance == "InverseManhattan":
                vec_clusters_densities = density_functions. \
                    inverse_manhattan_density(mat_membership, mat_entries,
                                              mat_cluster_centers, mat_cluster_entry_indexes)

            var_old_loss = var_new_loss
            # var_new_loss = sum(vec_clusters_densities) / var_init_count_clusters
            var_new_loss = (sum(vec_clusters_densities) / var_init_count_clusters) * loss_fix(var_init_count_clusters)
            # var_new_loss = sum(vec_clusters_densities) * loss_fix(var_init_count_clusters)

        var_old_loss = 0.0

        print("\n\ncount clusters", var_init_count_clusters)
        print("total loss", var_new_loss)
        vec_total_densities.append(var_new_loss)
        vec_clusters_count.append(var_init_count_clusters)

        var_max_loss_cluster_index = vec_clusters_densities.index(max(vec_clusters_densities))
        del mat_cluster_centers[var_max_loss_cluster_index]
        if "Mahalanobis" in distance:
            del ten_covariances[var_max_loss_cluster_index]

        var_init_count_clusters -= 1

        print("cluster losses:", vec_clusters_densities)
        vec_count_entries_in_clusters = []
        for x in mat_cluster_entry_indexes:
            vec_count_entries_in_clusters.append(len(x))
        print("count elems:", vec_count_entries_in_clusters)
        print("deleted cluster:", var_max_loss_cluster_index)

        if "Manhattan" in distance:
            mat_cluster_entry_indexes = entry_cluster_assignment. \
                manhattan_cluster_assignment(mat_entries, mat_cluster_centers)
        mat_cluster_entry_indexes = entry_cluster_assignment.balance_cluster_assignment(mat_cluster_entry_indexes)

    print("\n\ncount clusters", var_init_count_clusters)
    print("total loss", sum(vec_clusters_densities) / var_init_count_clusters)
    vec_total_densities.append(var_new_loss)
    vec_clusters_count.append(var_init_count_clusters)

    return clustering_results.ClusteringResults(vec_clusters_count=vec_clusters_count,
                                                vec_total_losses=vec_total_densities,
                                                mat_cluster_centers=mat_cluster_centers,
                                                mat_cluster_entry_indexes=mat_cluster_entry_indexes)

