import cluster_centers
import entry_cluster_assignment
import covariances
import memberships
import accuracies


def mahalanobis_clustering(mat_entries, var_count_clusters, vec_correct_entry_class):
    mat_cluster_centers = cluster_centers.random_cluster_centers(mat_entries, var_count_clusters)
    print("Initial cluster centers")
    print(mat_cluster_centers)
    mat_cluster_entry_indexes = entry_cluster_assignment.manhattan_cluster_assignment(mat_entries, mat_cluster_centers)
    print("Initial cluster entry indexes by manhattan cluster assignment")
    print(mat_cluster_entry_indexes)
    ten_covariances = covariances.cluster_covariances(mat_entries, mat_cluster_entry_indexes, mat_cluster_centers)
    print("Covariances")
    print(ten_covariances)

    mat_membership = memberships.mahalanobis_membership_matrix(mat_entries, mat_cluster_centers, ten_covariances)
    print("Membership")
    print(mat_membership)
    var_accuracy = accuracies.accuracy(mat_cluster_entry_indexes, vec_correct_entry_class)
    mat_confusion = accuracies.confusion_matrix(mat_cluster_entry_indexes, vec_correct_entry_class)

    for i in range(10):
        print("\n\n\nstep:", i, "accuracy:", var_accuracy)
        print("Confusion")
        for vec_confusion in mat_confusion:
            print(vec_confusion)
        cluster_centers.c_means_centers_moving(mat_cluster_centers, mat_entries, mat_membership,
                                               mat_cluster_entry_indexes)
        mat_membership = memberships.mahalanobis_membership_matrix(mat_entries, mat_cluster_centers, ten_covariances)
        print(mat_membership)
        ten_covariances = covariances.cluster_covariances(mat_entries, mat_cluster_entry_indexes, mat_cluster_centers)

        mat_cluster_entry_indexes = entry_cluster_assignment.\
            mahalanobis_cluster_assignment(mat_entries, mat_cluster_centers, ten_covariances)
        print(mat_cluster_entry_indexes)

        var_accuracy = accuracies.accuracy(mat_cluster_entry_indexes, vec_correct_entry_class)
        mat_confusion = accuracies.confusion_matrix(mat_cluster_entry_indexes, vec_correct_entry_class)


def cluster_evolve_clustering(mat_entries, var_count_clusters, vec_correct_entry_class):
    var_init_count_clusters = len(vec_correct_entry_class)
    mat_cluster_centers = cluster_centers.random_cluster_centers_from_entries(mat_entries, var_init_count_clusters)
    mat_cluster_entry_indexes = entry_cluster_assignment.manhattan_cluster_assignment(mat_entries, mat_cluster_centers)
    ten_covariances = covariances.cluster_covariances(mat_entries, mat_cluster_entry_indexes, mat_cluster_centers)
    mat_membership = memberships.mahalanobis_membership_matrix(mat_entries, mat_cluster_centers, ten_covariances)

