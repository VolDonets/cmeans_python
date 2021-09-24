import cluster_centers
import entry_cluster_assignment
import covariances
import memberships
import accuracies
import density_functions
import clustering_results


def clustering_by_manhattan_density(mat_entries, var_count_clusters, vec_correct_entry_class, var_init_count_clusters):
    mat_cluster_centers = cluster_centers.random_cluster_centers_from_entries(mat_entries, var_init_count_clusters)
    mat_cluster_entry_indexes = entry_cluster_assignment.manhattan_cluster_assignment(mat_entries, mat_cluster_centers)
    ten_covariances = covariances.cluster_covariances(mat_entries, mat_cluster_entry_indexes, mat_cluster_centers)
    mat_membership = memberships.mahalanobis_membership_matrix(mat_entries, mat_cluster_centers, ten_covariances)
    vec_clusters_densities = density_functions.manhattan_density(mat_membership, mat_entries, mat_cluster_centers,
                                                                 mat_cluster_entry_indexes)
    print(vec_clusters_densities)
    vec_total_densities = []
    vec_clusters_count = []

    # loop for removing useless cluster centers
    while var_init_count_clusters > var_count_clusters:
        print("\n\ncount clusters", len(mat_cluster_centers))
        print("total loss", sum(vec_clusters_densities))
        vec_total_densities.append(sum(vec_clusters_densities))
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
        else:
            var_max_loss_cluster_index = vec_clusters_densities.index(min(vec_clusters_densities))
            del mat_cluster_centers[var_max_loss_cluster_index]
        print("cluster losses:", vec_clusters_densities)
        vec_count_entries_in_clusters = []
        for x in mat_cluster_entry_indexes:
            vec_count_entries_in_clusters.append(len(x))
        print("count elems:", vec_count_entries_in_clusters)
        print("deleted cluster:", var_max_loss_cluster_index)

        mat_cluster_entry_indexes = entry_cluster_assignment.\
            mahalanobis_cluster_assignment(mat_entries, mat_cluster_centers, ten_covariances)
        ten_covariances = covariances.cluster_covariances(mat_entries, mat_cluster_entry_indexes, mat_cluster_centers)
        mat_membership = memberships.mahalanobis_membership_matrix(mat_entries, mat_cluster_centers, ten_covariances)
        vec_clusters_densities = density_functions.manhattan_density(mat_membership, mat_entries, mat_cluster_centers,
                                                                     mat_cluster_entry_indexes)
        var_init_count_clusters -= 1

        if var_init_count_clusters == var_count_clusters:
            vec_total_densities.append(sum(vec_clusters_densities))
            vec_clusters_count.append(var_init_count_clusters)

    var_accuracy = accuracies.accuracy(mat_cluster_entry_indexes, vec_correct_entry_class)
    mat_confusion = accuracies.confusion_matrix(mat_cluster_entry_indexes, vec_correct_entry_class)

    print("Accuracy:", var_accuracy)
    print("Confusion")
    for vec_confusion in mat_confusion:
        print(vec_confusion)

    print("clusters entries:")
    for vec_cluster_entry_indexes in mat_cluster_entry_indexes:
        print(vec_cluster_entry_indexes)
    return clustering_results.ClusteringResults(vec_clusters_count=vec_clusters_count,
                                                vec_total_losses=vec_total_densities,
                                                mat_cluster_centers=mat_cluster_centers,
                                                mat_cluster_entry_indexes=mat_cluster_entry_indexes)


def clustering_by_euclid_density(mat_entries, var_count_clusters, vec_correct_entry_class, var_init_count_clusters):
    mat_cluster_centers = cluster_centers.random_cluster_centers_from_entries(mat_entries, var_init_count_clusters)
    mat_cluster_entry_indexes = entry_cluster_assignment.manhattan_cluster_assignment(mat_entries, mat_cluster_centers)
    ten_covariances = covariances.cluster_covariances(mat_entries, mat_cluster_entry_indexes, mat_cluster_centers)
    mat_membership = memberships.mahalanobis_membership_matrix(mat_entries, mat_cluster_centers, ten_covariances)
    vec_clusters_densities = density_functions.euclid_density(mat_membership, mat_entries, mat_cluster_centers,
                                                                 mat_cluster_entry_indexes)
    print(vec_clusters_densities)
    vec_total_densities = []
    vec_clusters_count = []

    # loop for removing useless cluster centers
    while var_init_count_clusters > var_count_clusters:
        print("\n\ncount clusters", len(mat_cluster_centers))
        print("total loss", sum(vec_clusters_densities))
        vec_total_densities.append(sum(vec_clusters_densities))
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
        else:
            var_max_loss_cluster_index = vec_clusters_densities.index(min(vec_clusters_densities))
            del mat_cluster_centers[var_max_loss_cluster_index]
        print("cluster losses:", vec_clusters_densities)
        vec_count_entries_in_clusters = []
        for x in mat_cluster_entry_indexes:
            vec_count_entries_in_clusters.append(len(x))
        print("count elems:", vec_count_entries_in_clusters)
        print("deleted cluster:", var_max_loss_cluster_index)

        mat_cluster_entry_indexes = entry_cluster_assignment.\
            mahalanobis_cluster_assignment(mat_entries, mat_cluster_centers, ten_covariances)
        ten_covariances = covariances.cluster_covariances(mat_entries, mat_cluster_entry_indexes, mat_cluster_centers)
        mat_membership = memberships.mahalanobis_membership_matrix(mat_entries, mat_cluster_centers, ten_covariances)
        vec_clusters_densities = density_functions.euclid_density(mat_membership, mat_entries, mat_cluster_centers,
                                                                     mat_cluster_entry_indexes)
        var_init_count_clusters -= 1

        if var_init_count_clusters == var_count_clusters:
            vec_total_densities.append(sum(vec_clusters_densities))
            vec_clusters_count.append(var_init_count_clusters)

    var_accuracy = accuracies.accuracy(mat_cluster_entry_indexes, vec_correct_entry_class)
    mat_confusion = accuracies.confusion_matrix(mat_cluster_entry_indexes, vec_correct_entry_class)

    print("Accuracy:", var_accuracy)
    print("Confusion")
    for vec_confusion in mat_confusion:
        print(vec_confusion)

    print("clusters entries:")
    for vec_cluster_entry_indexes in mat_cluster_entry_indexes:
        print(vec_cluster_entry_indexes)
    return clustering_results.ClusteringResults(vec_clusters_count=vec_clusters_count,
                                                vec_total_losses=vec_total_densities,
                                                mat_cluster_centers=mat_cluster_centers,
                                                mat_cluster_entry_indexes=mat_cluster_entry_indexes)


def clustering_by_simple_mahalanobis_density(mat_entries, var_count_clusters, vec_correct_entry_class, 
                                             var_init_count_clusters):
    mat_cluster_centers = cluster_centers.random_cluster_centers_from_entries(mat_entries, var_init_count_clusters)
    mat_cluster_entry_indexes = entry_cluster_assignment.manhattan_cluster_assignment(mat_entries, mat_cluster_centers)
    ten_covariances = covariances.cluster_covariances(mat_entries, mat_cluster_entry_indexes, mat_cluster_centers)
    mat_membership = memberships.mahalanobis_membership_matrix(mat_entries, mat_cluster_centers, ten_covariances)
    vec_clusters_densities = density_functions.simple_mahalanobis_density(mat_entries, mat_cluster_centers,
                                                                          ten_covariances, mat_cluster_entry_indexes)
    print(vec_clusters_densities)
    vec_total_densities = []
    vec_clusters_count = []

    # loop for removing useless cluster centers
    while var_init_count_clusters > var_count_clusters:
        print("\n\ncount clusters", len(mat_cluster_centers))
        print("total loss", sum(vec_clusters_densities))
        vec_total_densities.append(sum(vec_clusters_densities))
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
        else:
            var_max_loss_cluster_index = vec_clusters_densities.index(min(vec_clusters_densities))
            del mat_cluster_centers[var_max_loss_cluster_index]
        print("cluster losses:", vec_clusters_densities)
        vec_count_entries_in_clusters = []
        for x in mat_cluster_entry_indexes:
            vec_count_entries_in_clusters.append(len(x))
        print("count elems:", vec_count_entries_in_clusters)
        print("deleted cluster:", var_max_loss_cluster_index)

        mat_cluster_entry_indexes = entry_cluster_assignment.\
            mahalanobis_cluster_assignment(mat_entries, mat_cluster_centers, ten_covariances)
        ten_covariances = covariances.cluster_covariances(mat_entries, mat_cluster_entry_indexes, mat_cluster_centers)
        mat_membership = memberships.mahalanobis_membership_matrix(mat_entries, mat_cluster_centers, ten_covariances)
        vec_clusters_densities = density_functions.simple_mahalanobis_density(mat_entries, mat_cluster_centers,
                                                                              ten_covariances,
                                                                              mat_cluster_entry_indexes)
        var_init_count_clusters -= 1

        if var_init_count_clusters == var_count_clusters:
            vec_total_densities.append(sum(vec_clusters_densities))
            vec_clusters_count.append(var_init_count_clusters)

    var_accuracy = accuracies.accuracy(mat_cluster_entry_indexes, vec_correct_entry_class)
    mat_confusion = accuracies.confusion_matrix(mat_cluster_entry_indexes, vec_correct_entry_class)

    print("Accuracy:", var_accuracy)
    print("Confusion")
    for vec_confusion in mat_confusion:
        print(vec_confusion)

    print("clusters entries:")
    for vec_cluster_entry_indexes in mat_cluster_entry_indexes:
        print(vec_cluster_entry_indexes)
    return clustering_results.ClusteringResults(vec_clusters_count=vec_clusters_count,
                                                vec_total_losses=vec_total_densities,
                                                mat_cluster_centers=mat_cluster_centers,
                                                mat_cluster_entry_indexes=mat_cluster_entry_indexes)


def clustering_by_mahalanobis_density(mat_entries, var_count_clusters, vec_correct_entry_class, var_init_count_clusters):
    mat_cluster_centers = cluster_centers.random_cluster_centers_from_entries(mat_entries, var_init_count_clusters)
    mat_cluster_entry_indexes = entry_cluster_assignment.manhattan_cluster_assignment(mat_entries, mat_cluster_centers)
    ten_covariances = covariances.cluster_covariances(mat_entries, mat_cluster_entry_indexes, mat_cluster_centers)
    mat_membership = memberships.mahalanobis_membership_matrix(mat_entries, mat_cluster_centers, ten_covariances)
    vec_clusters_densities = density_functions.mahalanobis_density(mat_membership, mat_entries, mat_cluster_centers,
                                                                   ten_covariances, mat_cluster_entry_indexes)
    print(vec_clusters_densities)
    vec_total_densities = []
    vec_clusters_count = []

    # loop for removing useless cluster centers
    while var_init_count_clusters > var_count_clusters:
        print("\n\ncount clusters", len(mat_cluster_centers))
        print("total loss", sum(vec_clusters_densities))
        vec_total_densities.append(sum(vec_clusters_densities))
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
        else:
            var_max_loss_cluster_index = vec_clusters_densities.index(min(vec_clusters_densities))
            del mat_cluster_centers[var_max_loss_cluster_index]
        print("cluster losses:", vec_clusters_densities)
        vec_count_entries_in_clusters = []
        for x in mat_cluster_entry_indexes:
            vec_count_entries_in_clusters.append(len(x))
        print("count elems:", vec_count_entries_in_clusters)
        print("deleted cluster:", var_max_loss_cluster_index)

        mat_cluster_entry_indexes = entry_cluster_assignment.\
            mahalanobis_cluster_assignment(mat_entries, mat_cluster_centers, ten_covariances)
        ten_covariances = covariances.cluster_covariances(mat_entries, mat_cluster_entry_indexes, mat_cluster_centers)
        mat_membership = memberships.mahalanobis_membership_matrix(mat_entries, mat_cluster_centers, ten_covariances)
        vec_clusters_densities = density_functions.mahalanobis_density(mat_membership, mat_entries, mat_cluster_centers,
                                                                       ten_covariances, mat_cluster_entry_indexes)
        var_init_count_clusters -= 1

        if var_init_count_clusters == var_count_clusters:
            vec_total_densities.append(sum(vec_clusters_densities))
            vec_clusters_count.append(var_init_count_clusters)

    var_accuracy = accuracies.accuracy(mat_cluster_entry_indexes, vec_correct_entry_class)
    mat_confusion = accuracies.confusion_matrix(mat_cluster_entry_indexes, vec_correct_entry_class)

    print("Accuracy:", var_accuracy)
    print("Confusion")
    for vec_confusion in mat_confusion:
        print(vec_confusion)

    print("clusters entries:")
    for vec_cluster_entry_indexes in mat_cluster_entry_indexes:
        print(vec_cluster_entry_indexes)
    return clustering_results.ClusteringResults(vec_clusters_count=vec_clusters_count,
                                                vec_total_losses=vec_total_densities,
                                                mat_cluster_centers=mat_cluster_centers,
                                                mat_cluster_entry_indexes=mat_cluster_entry_indexes)


def clustering_by_inverse_mahalanobis_density(mat_entries, var_count_clusters,
                                              vec_correct_entry_class, var_init_count_clusters):
    mat_cluster_centers = cluster_centers.random_cluster_centers_from_entries(mat_entries, var_init_count_clusters)
    mat_cluster_entry_indexes = entry_cluster_assignment.manhattan_cluster_assignment(mat_entries, mat_cluster_centers)
    ten_covariances = covariances.cluster_covariances(mat_entries, mat_cluster_entry_indexes, mat_cluster_centers)
    mat_membership = memberships.mahalanobis_membership_matrix(mat_entries, mat_cluster_centers, ten_covariances)
    vec_clusters_densities = density_functions.inverse_mahalanobis_density(mat_membership, mat_entries,
                                                                           mat_cluster_centers, ten_covariances,
                                                                           mat_cluster_entry_indexes)
    print(vec_clusters_densities)
    vec_total_densities = []
    vec_clusters_count = []

    # loop for removing useless cluster centers
    while var_init_count_clusters > var_count_clusters:
        print("\n\ncount clusters", len(mat_cluster_centers))
        print("total loss", sum(vec_clusters_densities))
        vec_total_densities.append(sum(vec_clusters_densities))
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
        else:
            var_max_loss_cluster_index = vec_clusters_densities.index(min(vec_clusters_densities))
            del mat_cluster_centers[var_max_loss_cluster_index]
        print("cluster losses:", vec_clusters_densities)
        vec_count_entries_in_clusters = []
        for x in mat_cluster_entry_indexes:
            vec_count_entries_in_clusters.append(len(x))
        print("count elems:", vec_count_entries_in_clusters)
        print("deleted cluster:", var_max_loss_cluster_index)

        mat_cluster_entry_indexes = entry_cluster_assignment.\
            mahalanobis_cluster_assignment(mat_entries, mat_cluster_centers, ten_covariances)
        ten_covariances = covariances.cluster_covariances(mat_entries, mat_cluster_entry_indexes, mat_cluster_centers)
        mat_membership = memberships.mahalanobis_membership_matrix(mat_entries, mat_cluster_centers, ten_covariances)
        vec_clusters_densities = density_functions.inverse_mahalanobis_density(mat_membership, mat_entries,
                                                                               mat_cluster_centers, ten_covariances,
                                                                               mat_cluster_entry_indexes)
        var_init_count_clusters -= 1

        if var_init_count_clusters == var_count_clusters:
            vec_total_densities.append(sum(vec_clusters_densities))
            vec_clusters_count.append(var_init_count_clusters)

    var_accuracy = accuracies.accuracy(mat_cluster_entry_indexes, vec_correct_entry_class)
    mat_confusion = accuracies.confusion_matrix(mat_cluster_entry_indexes, vec_correct_entry_class)

    print("Accuracy:", var_accuracy)
    print("Confusion")
    for vec_confusion in mat_confusion:
        print(vec_confusion)

    print("clusters entries:")
    for vec_cluster_entry_indexes in mat_cluster_entry_indexes:
        print(vec_cluster_entry_indexes)
    return clustering_results.ClusteringResults(vec_clusters_count=vec_clusters_count,
                                                vec_total_losses=vec_total_densities,
                                                mat_cluster_centers=mat_cluster_centers,
                                                mat_cluster_entry_indexes=mat_cluster_entry_indexes)
