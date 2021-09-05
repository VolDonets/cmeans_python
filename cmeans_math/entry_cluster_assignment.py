"""Cluster assignment

This script contains functions for entries assignment for each clusters (that means to do clustering)

Functions: 
    * manhattan_cluster_assignment
    * euclidean_cluster_assignment
    * mahalanobis_cluster_assignment
    * mahalanobis_cauchy_cluster_assignment
    * mahalanobis_own_old_distribution_cluster_assignment
"""

import distances
import distributions


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
