"""Cluster assignment

This script contains functions for entries assignment for each clusters (that means to do clustering)

Functions: 
    * manhattan_cluster_assignment
    * euclidean_cluster_assignment
    * mahalanobis_cluster_assignment
    * mahalanobis_cauchy_cluster_assignment
    * mahalanobis_own_old_distribution_cluster_assignment
"""
import random

import distances
import distributions
import divergences


def balance_cluster_assignment(mat_cluster_entry_indexes):
    # a bit ducking cracking
    # so if we have an empty cluster // less than 5 elements I'll separate too huge one
    while True:
        var_emp_cluster_inx = -1
        var_biggest_cluster_inx = 0
        for inx in range(len(mat_cluster_entry_indexes)):
            if len(mat_cluster_entry_indexes[inx]) < 2:
                var_emp_cluster_inx = inx
                break
        if var_emp_cluster_inx == -1:
            break
        for inx in range(len(mat_cluster_entry_indexes)):
            if len(mat_cluster_entry_indexes[inx]) > len(mat_cluster_entry_indexes[var_biggest_cluster_inx]):
                var_biggest_cluster_inx = inx

        var_count_moved_indexes = 5
        for i in range(var_count_moved_indexes):
            mat_cluster_entry_indexes[var_emp_cluster_inx].append(mat_cluster_entry_indexes[var_biggest_cluster_inx][0])
            del mat_cluster_entry_indexes[var_biggest_cluster_inx][0]
    return mat_cluster_entry_indexes


def manhattan_cluster_assignment(mat_entries, mat_cluster_centers):
    """
    :param mat_entries: vector of entry vectors
    :param mat_cluster_centers: vector of cluster centers vectors
    :return: matrix: vector of vectors with indexes 
           of entries (in mat_entries) for entries of each cluster
    """

    mat_cluster_entry_indexes = []
    for cl_inx in range(len(mat_cluster_centers)):
        mat_cluster_entry_indexes.append([])

    for entry_inx in range(len(mat_entries)):
        vec_distance_to_clusters = []
        for cl_inx in range(len(mat_cluster_centers)):
            vec_distance_to_clusters.append(distances.manhattan_distance(mat_entries[entry_inx],
                                                                         mat_cluster_centers[cl_inx]))
        var_selected_cluster = vec_distance_to_clusters.index(min(vec_distance_to_clusters))
        mat_cluster_entry_indexes[var_selected_cluster].append(entry_inx)
    return mat_cluster_entry_indexes


def euclidean_cluster_assignment(mat_entries, mat_cluster_centers):
    """
    :param mat_entries: vector of entry vectors
    :param mat_cluster_centers: vector of cluster centers vectors
    :return: matrix: vector of vectors with indexes
           of entries (in mat_entries) for entries of each cluster
    """

    mat_cluster_entry_indexes = []
    for cl_inx in range(len(mat_cluster_centers)):
        mat_cluster_entry_indexes.append([])

    for entry_inx in range(len(mat_entries)):
        vec_distance_to_clusters = []
        for cl_inx in range(len(mat_cluster_centers)):
            vec_distance_to_clusters.append(distances.euclidean_distance(mat_entries[entry_inx],
                                                                         mat_cluster_centers[cl_inx]))
        var_selected_cluster = vec_distance_to_clusters.index(min(vec_distance_to_clusters))
        mat_cluster_entry_indexes[var_selected_cluster].append(entry_inx)
    return mat_cluster_entry_indexes


def mahalanobis_cluster_assignment(mat_entries, mat_cluster_centers, ten_covariances):
    """
    :param mat_entries: vector of entry vectors
    :param mat_cluster_centers: vector of cluster centers vectors
    :param ten_covariances: vector of covariance matrices
    :return: matrix: vector of vectors with indexes
           of entries (in mat_entries) for entries of each cluster
    """

    mat_cluster_entry_indexes = []
    for cl_inx in range(len(mat_cluster_centers)):
        mat_cluster_entry_indexes.append([])

    for entry_inx in range(len(mat_entries)):
        vec_distance_to_clusters = []
        for cl_inx in range(len(mat_cluster_centers)):
            vec_distance_to_clusters.append(distances.mahalanobis_distance(mat_entries[entry_inx],
                                                                           mat_cluster_centers[cl_inx],
                                                                           ten_covariances[cl_inx]))
        var_selected_cluster = vec_distance_to_clusters.index(min(vec_distance_to_clusters))
        mat_cluster_entry_indexes[var_selected_cluster].append(entry_inx)
    return mat_cluster_entry_indexes


def mahalanobis_cauchy_cluster_assignment(mat_entries, mat_cluster_centers, ten_covariances):
    """
    :param mat_entries: vector of entry vectors
    :param mat_cluster_centers: vector of cluster centers vectors
    :param ten_covariances: vector of covariance matrices
    :return: matrix: vector of vectors with indexes
             of entries (in mat_entries) for entries of each cluster
    """

    mat_cluster_entry_indexes = []
    for cl_inx in range(len(mat_cluster_centers)):
        mat_cluster_entry_indexes.append([])

    for entry_inx in range(len(mat_entries)):
        vec_distance_to_clusters = []
        for cl_inx in range(len(mat_cluster_centers)):
            var_distance = distances.mahalanobis_distance(mat_entries[entry_inx],
                                                          mat_cluster_centers[cl_inx],
                                                          ten_covariances[cl_inx])
            var_distance = distributions.cauchy_distribution(1, var_distance)
            vec_distance_to_clusters.append(var_distance)
        var_selected_cluster = vec_distance_to_clusters.index(min(vec_distance_to_clusters))
        mat_cluster_entry_indexes[var_selected_cluster].append(entry_inx)
    return mat_cluster_entry_indexes


def mahalanobis_own_old_distribution_cluster_assignment(mat_entries, mat_cluster_centers, ten_covariances):
    """
    :param mat_entries: vector of entry vectors
    :param mat_cluster_centers: vector of cluster centers vectors
    :param ten_covariances: vector of covariance matrices
    :return: matrix: vector of vectors with indexes
             of entries (in mat_entries) for entries of each cluster
    """

    mat_cluster_entry_indexes = []
    for cl_inx in range(len(mat_cluster_centers)):
        mat_cluster_entry_indexes.append([])

    for entry_inx in range(len(mat_entries)):
        vec_distance_to_clusters = []
        for cl_inx in range(len(mat_cluster_centers)):
            var_distance = distances.mahalanobis_distance(mat_entries[entry_inx],
                                                          mat_cluster_centers[cl_inx],
                                                          ten_covariances[cl_inx])
            var_distance = distributions.own_old_distribution(1, 0.5, var_distance)
            vec_distance_to_clusters.append(var_distance)
        var_selected_cluster = vec_distance_to_clusters.index(min(vec_distance_to_clusters))
        mat_cluster_entry_indexes[var_selected_cluster].append(entry_inx)
    return mat_cluster_entry_indexes


def divergence_cluster_assignment(mat_entries, mat_cluster_centers, ten_covariances,
                                  divergence_func=divergences.kulback_leibler_divergence,
                                  basic_distance="Manhattan"):
    mat_cluster_entry_indexes = []
    for cl_inx in range(len(mat_cluster_centers)):
        mat_cluster_entry_indexes.append([])

    for entry_inx in range(len(mat_entries)):
        vec_distance_to_clusters = []
        for cl_inx in range(len(mat_cluster_centers)):
            var_distance = 0.0
            if basic_distance == "Manhattan":
                var_distance = distributions.\
                    cauchy_distribution(var_etta=1,
                                        var_distance=distances.manhattan_distance(mat_entries[entry_inx],
                                                                                  mat_cluster_centers[cl_inx]))
            elif basic_distance == "Mahalanobis":
                var_distance = distributions.\
                    cauchy_distribution(var_etta=1,
                                        var_distance=distances.mahalanobis_distance(mat_entries[entry_inx],
                                                                                    mat_cluster_centers[cl_inx]))
            vec_distance_to_clusters.append(var_distance)
        var_selected_cluster = vec_distance_to_clusters.index(max(vec_distance_to_clusters))
        # var_selected_cluster = vec_distance_to_clusters.index(min(vec_distance_to_clusters))
        mat_cluster_entry_indexes[var_selected_cluster].append(entry_inx)

    # a bit ducking cracking
    # so if we have an empty cluster // less than 5 elements I'll separate too huge one
    while True :
        var_emp_cluster_inx = -1
        var_biggest_cluster_inx = 0
        for inx in range(len(mat_cluster_entry_indexes)):
            if len(mat_cluster_entry_indexes[inx]) < 2:
                var_emp_cluster_inx = inx
                break
        if var_emp_cluster_inx == -1:
            break
        for inx in range(len(mat_cluster_entry_indexes)):
            if len(mat_cluster_entry_indexes[inx]) > len(mat_cluster_entry_indexes[var_biggest_cluster_inx]):
                var_biggest_cluster_inx = inx

        var_count_moved_indexes = 2
        for i in range(var_count_moved_indexes):
            mat_cluster_entry_indexes[var_emp_cluster_inx].append(mat_cluster_entry_indexes[var_biggest_cluster_inx][0])
            del mat_cluster_entry_indexes[var_biggest_cluster_inx][0]

    return mat_cluster_entry_indexes
