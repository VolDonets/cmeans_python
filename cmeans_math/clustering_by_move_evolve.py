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

# -1 - simple manhattan density
# 0 - manhattan density
# 1 - simple mahalanobis density
# 2 - inverse mahalanobis density
# 3 - inverse manhattan density
# 4 - mahalanobis density

CLUSTERING_TYPE = 2


def clustering_by_simple_mahalanobis_density(mat_entries, var_min_count_clusters, var_init_count_clusters,
                                             evolve_distance="Mahalanobis", clustering_type=CLUSTERING_TYPE):
    # generate random init cluster centers
    mat_cluster_centers = cluster_centers.random_cluster_centers(mat_entries=mat_entries,
                                                                 var_count_clusters=var_init_count_clusters)
    # assign entry to each cluster center
    mat_cluster_entry_indexes = entry_cluster_assignment.\
        manhattan_cluster_assignment(mat_entries=mat_entries, mat_cluster_centers=mat_cluster_centers)
    mat_cluster_entry_indexes = entry_cluster_assignment.balance_cluster_assignment(mat_cluster_entry_indexes)

    # get covariance matrices for each cluster
    ten_covariances = covariances.cluster_covariances(mat_entries, mat_cluster_entry_indexes, mat_cluster_centers)

    # get densities for the each cluster
    if clustering_type == -1:
        vec_clusters_densities = density_functions.simple_manhattan_density(mat_entries, mat_cluster_centers,
                                                                            mat_cluster_entry_indexes)
    elif clustering_type == 0:
        mat_membership = memberships.mahalanobis_membership_matrix(mat_entries, mat_cluster_centers, ten_covariances)
        vec_clusters_densities = density_functions.manhattan_density(mat_membership, mat_entries, mat_cluster_centers,
                                                                     mat_cluster_entry_indexes)
    elif clustering_type == 1:
        vec_clusters_densities = density_functions.simple_mahalanobis_density(mat_entries, mat_cluster_centers,
                                                                              ten_covariances,
                                                                              mat_cluster_entry_indexes)
    elif clustering_type == 2:
        mat_membership = memberships.mahalanobis_membership_matrix(mat_entries, mat_cluster_centers, ten_covariances)
        mat_membership = memberships.norm_membership_matrix_for_entries(mat_membership)
        vec_clusters_densities = density_functions.inverse_mahalanobis_density(mat_membership, mat_entries,
                                                                               mat_cluster_centers,
                                                                               ten_covariances,
                                                                               mat_cluster_entry_indexes)
    elif clustering_type == 3:
        mat_membership = memberships.manhattan_membership_matrix(mat_entries, mat_cluster_centers)
        mat_membership = memberships.norm_membership_matrix_for_entries(mat_membership)
        vec_clusters_densities = density_functions.inverse_manhattan_density(mat_membership, mat_entries,
                                                                             mat_cluster_centers,
                                                                             mat_cluster_entry_indexes)

    vec_total_densities = []
    vec_clusters_count = []

    var_old_loss = 0.0
    var_new_loss = sum(vec_clusters_densities) / var_init_count_clusters
    # var_new_loss = sum(vec_clusters_densities)

    # loop for removing useless cluster centers
    while var_init_count_clusters > var_min_count_clusters:

        # do some clusters moving
        i = 0
        while math.fabs(var_old_loss - var_new_loss) > MAX_LOSS_DELTA and i <= MAX_COUNT_TRAINING_STEPS:
            i += 1
            if evolve_distance == "Mahalanobis":
                mat_membership = memberships.mahalanobis_membership_matrix(mat_entries, mat_cluster_centers,
                                                                           ten_covariances)
            elif evolve_distance == "Manhattan":
                mat_membership = memberships.manhattan_membership_matrix(mat_entries, mat_cluster_centers)
            mat_membership = memberships.norm_membership_matrix_for_entries(mat_membership)

            cluster_centers.c_means_centers_moving(mat_cluster_centers, mat_entries, mat_membership,
                                                   mat_cluster_entry_indexes)

            mat_cluster_entry_indexes = entry_cluster_assignment. \
                mahalanobis_cluster_assignment(mat_entries, mat_cluster_centers, ten_covariances)
            mat_cluster_entry_indexes = entry_cluster_assignment.balance_cluster_assignment(mat_cluster_entry_indexes)

            ten_covariances = covariances.cluster_covariances(mat_entries, mat_cluster_entry_indexes,
                                                              mat_cluster_centers)

            if clustering_type == -1:
                vec_clusters_densities = density_functions.simple_manhattan_density(mat_entries, mat_cluster_centers,
                                                                                    mat_cluster_entry_indexes)
            elif clustering_type == 0:
                mat_membership = memberships.mahalanobis_membership_matrix(mat_entries, mat_cluster_centers,
                                                                           ten_covariances)
                vec_clusters_densities = density_functions.manhattan_density(mat_membership, mat_entries,
                                                                             mat_cluster_centers,
                                                                             mat_cluster_entry_indexes)
            elif clustering_type == 1:
                vec_clusters_densities = density_functions.simple_mahalanobis_density(mat_entries, mat_cluster_centers,
                                                                                      ten_covariances,
                                                                                      mat_cluster_entry_indexes)
            elif clustering_type == 2:
                mat_membership = memberships.mahalanobis_membership_matrix(mat_entries, mat_cluster_centers,
                                                                           ten_covariances)
                mat_membership = memberships.norm_membership_matrix_for_entries(mat_membership)
                vec_clusters_densities = density_functions.inverse_mahalanobis_density(mat_membership, mat_entries,
                                                                                       mat_cluster_centers,
                                                                                       ten_covariances,
                                                                                       mat_cluster_entry_indexes)
            elif clustering_type == 3:
                mat_membership = memberships.manhattan_membership_matrix(mat_entries, mat_cluster_centers)
                mat_membership = memberships.norm_membership_matrix_for_entries(mat_membership)
                vec_clusters_densities = density_functions.inverse_manhattan_density(mat_membership, mat_entries,
                                                                                     mat_cluster_centers,
                                                                                     mat_cluster_entry_indexes)

            var_old_loss = var_new_loss
            var_new_loss = sum(vec_clusters_densities) / var_init_count_clusters
            # var_new_loss = sum(vec_clusters_densities)

        var_old_loss = 0.0

        print("\n\ncount clusters", var_init_count_clusters)
        print("total loss", var_new_loss)
        vec_total_densities.append(var_new_loss)
        vec_clusters_count.append(var_init_count_clusters)

        # var_max_loss_cluster_index = 0
        # var_saved_len = 6
        # for j in range(len(mat_cluster_entry_indexes)):
        #     if len(mat_cluster_entry_indexes[j]) < 5:
        #         if var_saved_len > len(mat_cluster_entry_indexes[j]):
        #             var_saved_len = len(mat_cluster_entry_indexes[j])
        #             var_max_loss_cluster_index = j
        #
        # if var_saved_len < 6:
        #     del mat_cluster_centers[var_max_loss_cluster_index]
        #     del ten_covariances[var_max_loss_cluster_index]
        # else:
        var_max_loss_cluster_index = vec_clusters_densities.index(max(vec_clusters_densities))
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


def clustering_by_divergence_density(mat_entries, var_count_clusters, var_init_count_clusters,
                                     density_func="KulbackLeibler",
                                     evolve_distance="Mahalanobis",
                                     inner_evolve_distance="Manhattan"):
    # generate random init cluster centers
    mat_cluster_centers = cluster_centers.random_cluster_centers(mat_entries=mat_entries,
                                                                 var_count_clusters=var_init_count_clusters)
    # assign entry to each cluster center
    mat_cluster_entry_indexes = entry_cluster_assignment. \
        manhattan_cluster_assignment(mat_entries=mat_entries, mat_cluster_centers=mat_cluster_centers)

    # get covariance matrices for each cluster
    ten_covariances = covariances.cluster_covariances(mat_entries, mat_cluster_entry_indexes, mat_cluster_centers)

    # get densities for the each cluster
    vec_clusters_densities = []

    if density_func == "KulbackLeibler":
        vec_clusters_densities = density_functions.\
            new_divergence_density(mat_entries=mat_entries,
                                   mat_cluster_centers=mat_cluster_centers,
                                   ten_covariances=ten_covariances,
                                   mat_cluster_entry_indexes=mat_cluster_entry_indexes,
                                   get_divergence_membership_matrix_func=memberships.kulback_leibler_membership_matrix,
                                   basic_distance=inner_evolve_distance)
    elif density_func == "CrossEntropy":
        vec_clusters_densities = density_functions.\
            new_divergence_density(mat_entries=mat_entries,
                                   mat_cluster_centers=mat_cluster_centers,
                                   ten_covariances=ten_covariances,
                                   mat_cluster_entry_indexes=mat_cluster_entry_indexes,
                                   get_divergence_membership_matrix_func=memberships.cross_entropy_membership_matrix,
                                   basic_distance=inner_evolve_distance)

    vec_total_densities = []
    vec_clusters_count = []

    var_old_loss = 0.0
    var_new_loss = sum(vec_clusters_densities) / var_init_count_clusters
    # var_new_loss = sum(vec_clusters_densities)

    # loop for removing useless cluster centers
    is_optimized_last_clusters = False
    while var_init_count_clusters >= var_count_clusters:

        # do some clusters moving
        i = 0
        is_run_once = True
        while is_run_once:
            is_run_once = (math.fabs(var_old_loss - var_new_loss) > MAX_LOSS_DELTA and i <= MAX_COUNT_TRAINING_STEPS)
            i += 1
            if evolve_distance == "Mahalanobis":
                mat_membership = memberships.mahalanobis_membership_matrix(mat_entries, mat_cluster_centers,
                                                                           ten_covariances)
            elif evolve_distance == "Manhattan":
                mat_membership = memberships.manhattan_membership_matrix(mat_entries, mat_cluster_centers)
            elif evolve_distance == "KulbackLeibler":
                mat_membership = memberships.kulback_leibler_membership_matrix(mat_entries=mat_entries,
                                                                               mat_cluster_centers=mat_cluster_centers,
                                                                               ten_covariances=ten_covariances,
                                                                               basic_distance=inner_evolve_distance)
            elif evolve_distance == "CrossEntropy":
                mat_membership = memberships.cross_entropy_membership_matrix(mat_entries=mat_entries,
                                                                             mat_cluster_centers=mat_cluster_centers,
                                                                             ten_covariances=ten_covariances,
                                                                             basic_distance=inner_evolve_distance)
            mat_membership = memberships.norm_membership_matrix_for_entries(mat_membership)

            cluster_centers.c_means_centers_moving(mat_cluster_centers, mat_entries, mat_membership,
                                                   mat_cluster_entry_indexes)

            if evolve_distance == "Mahalanobis":
                mat_cluster_entry_indexes = entry_cluster_assignment. \
                    mahalanobis_cluster_assignment(mat_entries, mat_cluster_centers, ten_covariances)
            elif evolve_distance == "Manhattan":
                mat_cluster_entry_indexes = entry_cluster_assignment. \
                    manhattan_cluster_assignment(mat_entries, mat_cluster_centers)
            elif evolve_distance == "KulbackLeibler":
                mat_cluster_entry_indexes = entry_cluster_assignment.\
                    divergence_cluster_assignment(mat_entries=mat_entries,
                                                  mat_cluster_centers=mat_cluster_centers,
                                                  ten_covariances=ten_covariances,
                                                  divergence_func=divergences.kulback_leibler_divergence,
                                                  basic_distance=inner_evolve_distance)
            elif evolve_distance == "CrossEntropy":
                mat_cluster_entry_indexes = entry_cluster_assignment. \
                    divergence_cluster_assignment(mat_entries=mat_entries,
                                                  mat_cluster_centers=mat_cluster_centers,
                                                  ten_covariances=ten_covariances,
                                                  divergence_func=divergences.cross_entropy,
                                                  basic_distance=inner_evolve_distance)

            mat_cluster_entry_indexes = entry_cluster_assignment.balance_cluster_assignment(mat_cluster_entry_indexes)

            ten_covariances = covariances.cluster_covariances(mat_entries, mat_cluster_entry_indexes,
                                                              mat_cluster_centers)

            if density_func == "KulbackLeibler":
                vec_clusters_densities = density_functions.\
                    new_divergence_density(mat_entries=mat_entries,
                                           mat_cluster_centers=mat_cluster_centers,
                                           ten_covariances=ten_covariances,
                                           mat_cluster_entry_indexes=mat_cluster_entry_indexes,
                                           get_divergence_membership_matrix_func=memberships.kulback_leibler_membership_matrix,
                                           basic_distance=inner_evolve_distance)
            elif density_func == "CrossEntropy":
                vec_clusters_densities = density_functions.\
                    new_divergence_density(mat_entries=mat_entries,
                                           mat_cluster_centers=mat_cluster_centers,
                                           ten_covariances=ten_covariances,
                                           mat_cluster_entry_indexes=mat_cluster_entry_indexes,
                                           get_divergence_membership_matrix_func=memberships.cross_entropy_membership_matrix,
                                           basic_distance=inner_evolve_distance)

            var_old_loss = var_new_loss
            var_new_loss = sum(vec_clusters_densities) / var_init_count_clusters
            # var_new_loss = sum(vec_clusters_densities)

        var_old_loss = 0.0

        print("\n\ncount clusters", var_init_count_clusters)
        print("total loss", var_new_loss)
        vec_total_densities.append(var_new_loss)
        vec_clusters_count.append(var_init_count_clusters)

        if var_init_count_clusters != var_count_clusters:
            var_max_loss_cluster_index = vec_clusters_densities.index(max(vec_clusters_densities))
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
        mat_cluster_entry_indexes = entry_cluster_assignment.balance_cluster_assignment(mat_cluster_entry_indexes)
        ten_covariances = covariances.cluster_covariances(mat_entries, mat_cluster_entry_indexes, mat_cluster_centers)

        if is_optimized_last_clusters:
            break
        if var_init_count_clusters == var_count_clusters:
            is_optimized_last_clusters = True

    # print("\n\ncount clusters", var_init_count_clusters)
    # print("total loss", sum(vec_clusters_densities) / var_init_count_clusters)
    # vec_total_densities.append(var_new_loss)
    # vec_clusters_count.append(var_init_count_clusters)

    return clustering_results.ClusteringResults(vec_clusters_count=vec_clusters_count,
                                                vec_total_losses=vec_total_densities,
                                                mat_cluster_centers=mat_cluster_centers,
                                                mat_cluster_entry_indexes=mat_cluster_entry_indexes)
    return 0


def move_clustering_divergence_loss(mat_entries, var_count_clusters,
                                    mat_initial_cluster_centers=[],
                                    divergence_func=divergences.kulback_leibler_divergence,
                                    distance="Mahalanobis"):
    vec_steps = []
    vec_losses = []
    mat_cluster_centers = []

    if len(mat_initial_cluster_centers) == 0:
        mat_cluster_centers = cluster_centers.random_cluster_centers(mat_entries, var_count_clusters)
    else:
        mat_cluster_centers = mat_initial_cluster_centers
    mat_cluster_entry_indexes = entry_cluster_assignment.manhattan_cluster_assignment(mat_entries, mat_cluster_centers)
    ten_covariances = covariances.cluster_covariances(mat_entries, mat_cluster_entry_indexes, mat_cluster_centers)

    mat_membership = memberships.mahalanobis_membership_matrix(mat_entries, mat_cluster_centers, ten_covariances)
    mat_membership = memberships.norm_membership_matrix_for_entries(mat_membership)
    var_loss = loss_functions.inverse_mahalanobis_loss(mat_membership, mat_entries, mat_cluster_centers, ten_covariances)
    var_old_loss = -1.0
    i = 0

    cluster_centers.c_means_centers_moving(mat_cluster_centers, mat_entries, mat_membership,
                                           mat_cluster_entry_indexes)
    mat_old_membership = mat_membership
    mat_membership = memberships.mahalanobis_membership_matrix(mat_entries, mat_cluster_centers, ten_covariances)
    mat_membership = memberships.norm_membership_matrix_for_entries(mat_membership)
    mat_divergences = divergences.divergence(mat_membership, mat_old_membership,
                                             divergence_func=divergence_func)

    ten_covariances = covariances.cluster_covariances(mat_entries, mat_cluster_entry_indexes, mat_cluster_centers)

    if distance == "Mahalanobis":
        mat_cluster_entry_indexes = entry_cluster_assignment. \
            mahalanobis_cluster_assignment(mat_entries, mat_cluster_centers, ten_covariances)
    elif distance == "Manhattan":
        mat_cluster_entry_indexes = entry_cluster_assignment. \
            manhattan_cluster_assignment(mat_entries, mat_cluster_centers)

    var_old_loss = var_loss
    var_loss = loss_functions.divergence_loss(mat_divergences)

    ##########################

    while math.fabs(var_old_loss - var_loss) > MAX_LOSS_DELTA and i <= MAX_COUNT_TRAINING_STEPS:
        print("\n\n\nstep:", i, "loss:", var_loss)
        vec_steps.append(i)
        vec_losses.append(var_loss)

        cluster_centers.c_means_centers_moving(mat_cluster_centers, mat_entries, mat_membership,
                                               mat_cluster_entry_indexes)
        mat_old_membership = mat_membership
        mat_membership = memberships.mahalanobis_membership_matrix(mat_entries, mat_cluster_centers, ten_covariances)
        mat_membership = memberships.norm_membership_matrix_for_entries(mat_membership)
        mat_divergences = divergences.divergence(mat_membership, mat_old_membership,
                                                 divergence_func=divergence_func)

        ten_covariances = covariances.cluster_covariances(mat_entries, mat_cluster_entry_indexes, mat_cluster_centers)

        if distance == "Mahalanobis":
            mat_cluster_entry_indexes = entry_cluster_assignment. \
                mahalanobis_cluster_assignment(mat_entries, mat_cluster_centers, ten_covariances)
        elif distance == "Manhattan":
            mat_cluster_entry_indexes = entry_cluster_assignment. \
                manhattan_cluster_assignment(mat_entries, mat_cluster_centers)

        i += 1
        var_old_loss = var_loss
        var_loss = loss_functions.divergence_loss(mat_divergences)

    return clustering_results.ClusteringResults(mat_cluster_centers=mat_cluster_centers,
                                                mat_cluster_entry_indexes=mat_cluster_entry_indexes,
                                                vec_step_number=vec_steps,
                                                vec_total_losses=vec_losses)


def move_evolve_clustering_divergence_loss(mat_entries, var_init_count_clusters, var_min_count_clusters,
                                           divergence_func=divergences.kulback_leibler_divergence,
                                           distance="Mahalanobis"):
    clustering_result = move_clustering_divergence_loss(mat_entries=mat_entries,
                                                        var_count_clusters=var_init_count_clusters,
                                                        divergence_func=divergence_func,
                                                        distance=distance)
    vec_steps = []
    vec_losses = []

    while var_init_count_clusters >= var_min_count_clusters:
        vec_steps.append(var_init_count_clusters)
        vec_losses.append(sum(clustering_result.vec_total_losses) / len(clustering_result.vec_total_losses))
        var_init_count_clusters -= 1

        # remove one cluster
        vec_cluster_size = []
        for i in range(len(clustering_result.mat_cluster_entry_indexes)):
            vec_cluster_size.append(len(clustering_result.mat_cluster_entry_indexes[i]))
        var_rm_inx = vec_cluster_size.index(max(vec_cluster_size))
        del clustering_result.mat_cluster_centers[var_rm_inx]

        clustering_result = move_clustering_divergence_loss(mat_entries=mat_entries,
                                                            var_count_clusters=var_init_count_clusters,
                                                            mat_initial_cluster_centers=clustering_result.mat_cluster_centers,
                                                            divergence_func=divergence_func,
                                                            distance=distance)

    return clustering_results.ClusteringResults(vec_clusters_count=vec_steps,
                                                vec_total_losses=vec_losses,
                                                mat_cluster_centers=clustering_result.mat_cluster_centers,
                                                mat_cluster_entry_indexes=clustering_result.mat_cluster_entry_indexes)

