import math

import cluster_centers
import entry_cluster_assignment
import covariances
import memberships
import accuracies
import clustering_results
import loss_functions
import divergences


def clustering_without_loss(mat_entries, var_count_clusters,
                            vec_correct_entry_class, var_count_studying_steps=10):
    mat_cluster_centers = cluster_centers.random_cluster_centers(mat_entries, var_count_clusters)
    mat_cluster_entry_indexes = entry_cluster_assignment.manhattan_cluster_assignment(mat_entries, mat_cluster_centers)
    ten_covariances = covariances.cluster_covariances(mat_entries, mat_cluster_entry_indexes, mat_cluster_centers)

    print("Initial cluster centers")
    print(mat_cluster_centers)
    print("Initial cluster entry indexes by manhattan cluster assignment")
    print(mat_cluster_entry_indexes)
    print("Covariances")
    print(ten_covariances)

    mat_membership = memberships.mahalanobis_membership_matrix(mat_entries, mat_cluster_centers, ten_covariances)
    mat_membership = memberships.norm_membership_matrix_for_entries(mat_membership)
    var_accuracy = accuracies.accuracy(mat_cluster_entry_indexes, vec_correct_entry_class)
    mat_confusion = accuracies.confusion_matrix(mat_cluster_entry_indexes, vec_correct_entry_class)

    print("Membership")
    print(mat_membership)

    for i in range(var_count_studying_steps):
        print("\n\n\nstep:", i, "accuracy:", var_accuracy)
        print("Confusion")
        for vec_confusion in mat_confusion:
            print(vec_confusion)

        cluster_centers.c_means_centers_moving(mat_cluster_centers, mat_entries, mat_membership,
                                               mat_cluster_entry_indexes)
        mat_membership = memberships.mahalanobis_membership_matrix(mat_entries, mat_cluster_centers, ten_covariances)
        mat_membership = memberships.norm_membership_matrix_for_entries(mat_membership)
        ten_covariances = covariances.cluster_covariances(mat_entries, mat_cluster_entry_indexes, mat_cluster_centers)

        mat_cluster_entry_indexes = entry_cluster_assignment.\
            mahalanobis_cluster_assignment(mat_entries, mat_cluster_centers, ten_covariances)

        var_accuracy = accuracies.accuracy(mat_cluster_entry_indexes, vec_correct_entry_class)
        mat_confusion = accuracies.confusion_matrix(mat_cluster_entry_indexes, vec_correct_entry_class)

        print(mat_membership)
        print(mat_cluster_entry_indexes)

    return clustering_results.ClusteringResults(mat_cluster_centers=mat_cluster_centers,
                                                mat_cluster_entry_indexes=mat_cluster_entry_indexes)


def clustering_manhattan_loss(mat_entries, var_count_clusters,
                              vec_correct_entry_class):
    vec_steps = []
    vec_losses = []

    mat_cluster_centers = cluster_centers.random_cluster_centers(mat_entries, var_count_clusters)
    mat_cluster_entry_indexes = entry_cluster_assignment.manhattan_cluster_assignment(mat_entries, mat_cluster_centers)
    ten_covariances = covariances.cluster_covariances(mat_entries, mat_cluster_entry_indexes, mat_cluster_centers)

    print("Initial cluster centers")
    print(mat_cluster_centers)
    print("Initial cluster entry indexes by manhattan cluster assignment")
    print(mat_cluster_entry_indexes)
    print("Covariances")
    print(ten_covariances)

    mat_membership = memberships.mahalanobis_membership_matrix(mat_entries, mat_cluster_centers, ten_covariances)
    mat_membership = memberships.norm_membership_matrix_for_entries(mat_membership)
    var_accuracy = accuracies.accuracy(mat_cluster_entry_indexes, vec_correct_entry_class)
    mat_confusion = accuracies.confusion_matrix(mat_cluster_entry_indexes, vec_correct_entry_class)
    var_loss = loss_functions.manhattan_loss(mat_membership, mat_entries, mat_cluster_centers)
    var_old_loss = -1.0

    print("Membership")
    print(mat_membership)

    i = 0
    while math.fabs(var_old_loss - var_loss) > 5:
        print("\n\n\nstep:", i, "accuracy:", var_accuracy, "loss:", var_loss)
        vec_steps.append(i)
        vec_losses.append(var_loss)
        print("Confusion")
        for vec_confusion in mat_confusion:
            print(vec_confusion)

        cluster_centers.c_means_centers_moving(mat_cluster_centers, mat_entries, mat_membership,
                                               mat_cluster_entry_indexes)
        mat_membership = memberships.mahalanobis_membership_matrix(mat_entries, mat_cluster_centers, ten_covariances)
        mat_membership = memberships.norm_membership_matrix_for_entries(mat_membership)
        ten_covariances = covariances.cluster_covariances(mat_entries, mat_cluster_entry_indexes, mat_cluster_centers)

        mat_cluster_entry_indexes = entry_cluster_assignment.\
            mahalanobis_cluster_assignment(mat_entries, mat_cluster_centers, ten_covariances)

        var_accuracy = accuracies.accuracy(mat_cluster_entry_indexes, vec_correct_entry_class)
        mat_confusion = accuracies.confusion_matrix(mat_cluster_entry_indexes, vec_correct_entry_class)

        i += 1
        var_old_loss = var_loss
        var_loss = loss_functions.manhattan_loss(mat_membership, mat_entries, mat_cluster_centers)

        print(mat_membership)
        print(mat_cluster_entry_indexes)

    return clustering_results.ClusteringResults(mat_cluster_centers=mat_cluster_centers,
                                                mat_cluster_entry_indexes=mat_cluster_entry_indexes,
                                                vec_step_number=vec_steps,
                                                vec_total_losses=vec_losses)


def clustering_euclid_loss(mat_entries, var_count_clusters,
                           vec_correct_entry_class):
    vec_steps = []
    vec_losses = []

    mat_cluster_centers = cluster_centers.random_cluster_centers(mat_entries, var_count_clusters)
    mat_cluster_entry_indexes = entry_cluster_assignment.manhattan_cluster_assignment(mat_entries, mat_cluster_centers)
    ten_covariances = covariances.cluster_covariances(mat_entries, mat_cluster_entry_indexes, mat_cluster_centers)

    print("Initial cluster centers")
    print(mat_cluster_centers)
    print("Initial cluster entry indexes by manhattan cluster assignment")
    print(mat_cluster_entry_indexes)
    print("Covariances")
    print(ten_covariances)

    mat_membership = memberships.mahalanobis_membership_matrix(mat_entries, mat_cluster_centers, ten_covariances)
    mat_membership = memberships.norm_membership_matrix_for_entries(mat_membership)
    var_accuracy = accuracies.accuracy(mat_cluster_entry_indexes, vec_correct_entry_class)
    mat_confusion = accuracies.confusion_matrix(mat_cluster_entry_indexes, vec_correct_entry_class)
    var_loss = loss_functions.euclid_loss(mat_membership, mat_entries, mat_cluster_centers)
    var_old_loss = -1.0

    print("Membership")
    print(mat_membership)

    i = 0
    while math.fabs(var_old_loss - var_loss) > 5:
        print("\n\n\nstep:", i, "accuracy:", var_accuracy, "loss:", var_loss)
        vec_steps.append(i)
        vec_losses.append(var_loss)
        print("Confusion")
        for vec_confusion in mat_confusion:
            print(vec_confusion)

        cluster_centers.c_means_centers_moving(mat_cluster_centers, mat_entries, mat_membership,
                                               mat_cluster_entry_indexes)
        mat_membership = memberships.mahalanobis_membership_matrix(mat_entries, mat_cluster_centers, ten_covariances)
        mat_membership = memberships.norm_membership_matrix_for_entries(mat_membership)
        ten_covariances = covariances.cluster_covariances(mat_entries, mat_cluster_entry_indexes, mat_cluster_centers)

        mat_cluster_entry_indexes = entry_cluster_assignment.\
            mahalanobis_cluster_assignment(mat_entries, mat_cluster_centers, ten_covariances)

        var_accuracy = accuracies.accuracy(mat_cluster_entry_indexes, vec_correct_entry_class)
        mat_confusion = accuracies.confusion_matrix(mat_cluster_entry_indexes, vec_correct_entry_class)

        i += 1
        var_old_loss = var_loss
        var_loss = loss_functions.euclid_loss(mat_membership, mat_entries, mat_cluster_centers)

        print(mat_membership)
        print(mat_cluster_entry_indexes)

        if i > 30:
            break

    return clustering_results.ClusteringResults(mat_cluster_centers=mat_cluster_centers,
                                                mat_cluster_entry_indexes=mat_cluster_entry_indexes,
                                                vec_step_number=vec_steps,
                                                vec_total_losses=vec_losses)


def clustering_mahalanobis_loss(mat_entries, var_count_clusters,
                                vec_correct_entry_class):
    vec_steps = []
    vec_losses = []

    mat_cluster_centers = cluster_centers.random_cluster_centers(mat_entries, var_count_clusters)
    mat_cluster_entry_indexes = entry_cluster_assignment.manhattan_cluster_assignment(mat_entries, mat_cluster_centers)
    ten_covariances = covariances.cluster_covariances(mat_entries, mat_cluster_entry_indexes, mat_cluster_centers)

    print("Initial cluster centers")
    print(mat_cluster_centers)
    print("Initial cluster entry indexes by manhattan cluster assignment")
    print(mat_cluster_entry_indexes)
    print("Covariances")
    print(ten_covariances)

    mat_membership = memberships.mahalanobis_membership_matrix(mat_entries, mat_cluster_centers, ten_covariances)
    mat_membership = memberships.norm_membership_matrix_for_entries(mat_membership)
    var_accuracy = accuracies.accuracy(mat_cluster_entry_indexes, vec_correct_entry_class)
    mat_confusion = accuracies.confusion_matrix(mat_cluster_entry_indexes, vec_correct_entry_class)
    var_loss = loss_functions.mahalanobis_loss(mat_membership, mat_entries, mat_cluster_centers, ten_covariances)
    var_old_loss = -1.0

    print("Membership")
    print(mat_membership)

    i = 0
    while math.fabs(var_old_loss - var_loss) > 5:
        print("\n\n\nstep:", i, "accuracy:", var_accuracy, "loss:", var_loss)
        vec_steps.append(i)
        vec_losses.append(var_loss)
        print("Confusion")
        for vec_confusion in mat_confusion:
            print(vec_confusion)

        cluster_centers.c_means_centers_moving(mat_cluster_centers, mat_entries, mat_membership,
                                               mat_cluster_entry_indexes)
        mat_membership = memberships.mahalanobis_membership_matrix(mat_entries, mat_cluster_centers, ten_covariances)
        mat_membership = memberships.norm_membership_matrix_for_entries(mat_membership)
        ten_covariances = covariances.cluster_covariances(mat_entries, mat_cluster_entry_indexes, mat_cluster_centers)

        mat_cluster_entry_indexes = entry_cluster_assignment.\
            mahalanobis_cluster_assignment(mat_entries, mat_cluster_centers, ten_covariances)

        var_accuracy = accuracies.accuracy(mat_cluster_entry_indexes, vec_correct_entry_class)
        mat_confusion = accuracies.confusion_matrix(mat_cluster_entry_indexes, vec_correct_entry_class)

        i += 1
        var_old_loss = var_loss
        var_loss = loss_functions.mahalanobis_loss(mat_membership, mat_entries, mat_cluster_centers, ten_covariances)

        print(mat_membership)
        print(mat_cluster_entry_indexes)

        if i > 30:
            break

    return clustering_results.ClusteringResults(mat_cluster_centers=mat_cluster_centers,
                                                mat_cluster_entry_indexes=mat_cluster_entry_indexes,
                                                vec_step_number=vec_steps,
                                                vec_total_losses=vec_losses)


def clustering_inverse_mahalanobis_loss(mat_entries, var_count_clusters,
                                        vec_correct_entry_class):
    vec_steps = []
    vec_losses = []

    mat_cluster_centers = cluster_centers.random_cluster_centers(mat_entries, var_count_clusters)
    mat_cluster_entry_indexes = entry_cluster_assignment.manhattan_cluster_assignment(mat_entries, mat_cluster_centers)
    ten_covariances = covariances.cluster_covariances(mat_entries, mat_cluster_entry_indexes, mat_cluster_centers)

    print("Initial cluster centers")
    print(mat_cluster_centers)
    print("Initial cluster entry indexes by manhattan cluster assignment")
    print(mat_cluster_entry_indexes)
    print("Covariances")
    print(ten_covariances)

    mat_membership = memberships.mahalanobis_membership_matrix(mat_entries, mat_cluster_centers, ten_covariances)
    mat_membership = memberships.norm_membership_matrix_for_entries(mat_membership)
    var_accuracy = accuracies.accuracy(mat_cluster_entry_indexes, vec_correct_entry_class)
    mat_confusion = accuracies.confusion_matrix(mat_cluster_entry_indexes, vec_correct_entry_class)
    var_loss = loss_functions.inverse_mahalanobis_loss(mat_membership, mat_entries, mat_cluster_centers, ten_covariances)
    var_old_loss = -1.0

    print("Membership")
    print(mat_membership)

    i = 0
    while math.fabs(var_old_loss - var_loss) > 5:
        print("\n\n\nstep:", i, "accuracy:", var_accuracy, "loss:", var_loss)
        vec_steps.append(i)
        vec_losses.append(var_loss)
        print("Confusion")
        for vec_confusion in mat_confusion:
            print(vec_confusion)

        cluster_centers.c_means_centers_moving(mat_cluster_centers, mat_entries, mat_membership,
                                               mat_cluster_entry_indexes)
        mat_membership = memberships.mahalanobis_membership_matrix(mat_entries, mat_cluster_centers, ten_covariances)
        mat_membership = memberships.norm_membership_matrix_for_entries(mat_membership)
        ten_covariances = covariances.cluster_covariances(mat_entries, mat_cluster_entry_indexes, mat_cluster_centers)

        mat_cluster_entry_indexes = entry_cluster_assignment.\
            mahalanobis_cluster_assignment(mat_entries, mat_cluster_centers, ten_covariances)

        var_accuracy = accuracies.accuracy(mat_cluster_entry_indexes, vec_correct_entry_class)
        mat_confusion = accuracies.confusion_matrix(mat_cluster_entry_indexes, vec_correct_entry_class)

        i += 1
        var_old_loss = var_loss
        var_loss = loss_functions.inverse_mahalanobis_loss(mat_membership, mat_entries, mat_cluster_centers, ten_covariances)

        print(mat_membership)
        print(mat_cluster_entry_indexes)

        if i > 30:
            break

    return clustering_results.ClusteringResults(mat_cluster_centers=mat_cluster_centers,
                                                mat_cluster_entry_indexes=mat_cluster_entry_indexes,
                                                vec_step_number=vec_steps,
                                                vec_total_losses=vec_losses)


def clustering_divergence_loss(mat_entries, var_count_clusters,
                               vec_correct_entry_class):
    vec_steps = []
    vec_losses = []

    mat_cluster_centers = cluster_centers.random_cluster_centers(mat_entries, var_count_clusters)
    mat_cluster_entry_indexes = entry_cluster_assignment.manhattan_cluster_assignment(mat_entries, mat_cluster_centers)
    ten_covariances = covariances.cluster_covariances(mat_entries, mat_cluster_entry_indexes, mat_cluster_centers)

    print("Initial cluster centers")
    print(mat_cluster_centers)
    print("Initial cluster entry indexes by manhattan cluster assignment")
    print(mat_cluster_entry_indexes)
    print("Covariances")
    print(ten_covariances)

    mat_membership = memberships.mahalanobis_membership_matrix(mat_entries, mat_cluster_centers, ten_covariances)
    mat_membership = memberships.norm_membership_matrix_for_entries(mat_membership)
    var_accuracy = accuracies.accuracy(mat_cluster_entry_indexes, vec_correct_entry_class)
    mat_confusion = accuracies.confusion_matrix(mat_cluster_entry_indexes, vec_correct_entry_class)
    var_loss = loss_functions.inverse_mahalanobis_loss(mat_membership, mat_entries, mat_cluster_centers, ten_covariances)
    var_old_loss = -1.0
    i = 0

    print("Membership")
    print(mat_membership)

    ####################################

    print("\n\n\nstep:", i, "accuracy:", var_accuracy, "loss:", var_loss)
    print("Confusion")
    for vec_confusion in mat_confusion:
        print(vec_confusion)

    cluster_centers.c_means_centers_moving(mat_cluster_centers, mat_entries, mat_membership,
                                           mat_cluster_entry_indexes)
    mat_old_membership = mat_membership
    mat_membership = memberships.mahalanobis_membership_matrix(mat_entries, mat_cluster_centers, ten_covariances)
    mat_membership = memberships.norm_membership_matrix_for_entries(mat_membership)
    mat_divergences = divergences.divergence(mat_membership, mat_old_membership)

    ten_covariances = covariances.cluster_covariances(mat_entries, mat_cluster_entry_indexes, mat_cluster_centers)

    mat_cluster_entry_indexes = entry_cluster_assignment. \
        mahalanobis_cluster_assignment(mat_entries, mat_cluster_centers, ten_covariances)

    var_accuracy = accuracies.accuracy(mat_cluster_entry_indexes, vec_correct_entry_class)
    mat_confusion = accuracies.confusion_matrix(mat_cluster_entry_indexes, vec_correct_entry_class)

    var_old_loss = var_loss
    var_loss = loss_functions.divergence_loss(mat_divergences)

    print(mat_membership)
    print(mat_cluster_entry_indexes)

    ##########################

    while math.fabs(var_old_loss - var_loss) > 0.1:
        print("\n\n\nstep:", i, "accuracy:", var_accuracy, "loss:", var_loss)
        vec_steps.append(i)
        vec_losses.append(var_loss)
        print("Confusion")
        for vec_confusion in mat_confusion:
            print(vec_confusion)

        cluster_centers.c_means_centers_moving(mat_cluster_centers, mat_entries, mat_membership,
                                               mat_cluster_entry_indexes)
        mat_old_membership = mat_membership
        mat_membership = memberships.mahalanobis_membership_matrix(mat_entries, mat_cluster_centers, ten_covariances)
        mat_membership = memberships.norm_membership_matrix_for_entries(mat_membership)
        mat_divergences = divergences.divergence(mat_membership, mat_old_membership)

        ten_covariances = covariances.cluster_covariances(mat_entries, mat_cluster_entry_indexes, mat_cluster_centers)

        mat_cluster_entry_indexes = entry_cluster_assignment.\
            mahalanobis_cluster_assignment(mat_entries, mat_cluster_centers, ten_covariances)

        var_accuracy = accuracies.accuracy(mat_cluster_entry_indexes, vec_correct_entry_class)
        mat_confusion = accuracies.confusion_matrix(mat_cluster_entry_indexes, vec_correct_entry_class)

        i += 1
        var_old_loss = var_loss
        var_loss = loss_functions.divergence_loss(mat_divergences)

        print(mat_membership)
        print(mat_cluster_entry_indexes)

        if i > 30:
            break

    return clustering_results.ClusteringResults(mat_cluster_centers=mat_cluster_centers,
                                                mat_cluster_entry_indexes=mat_cluster_entry_indexes,
                                                vec_step_number=vec_steps,
                                                vec_total_losses=vec_losses)
