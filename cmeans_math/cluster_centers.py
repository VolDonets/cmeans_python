"""Cluster centers processing

This script contains functions for generating random cluster centers or for moving cluster centers

Functions:
    * random_cluster_centers - just generate random cluster centers
    * random_cluster_centers_from_entries - takes random entries as cluster centers
    * k_means_centers_moving - moves cluster centers by k-means algorithm
    * c_means_centers_moving - moves cluster centers by c-means algorithm
"""

import random


def random_cluster_centers(mat_entries, var_count_clusters):
    """
    :param mat_entries: vector of entry vectors
    :param var_count_clusters: - count clusters
    :return: vector of cluster centers vectors
    """

    mat_cluster_centers = []
    vec_min_entry = []
    vec_max_entry = []
    for j in range(len(mat_entries[0])):
        vec_min_entry.append(mat_entries[0][j])
        vec_max_entry.append(mat_entries[0][j])

    for i in range(1, len(mat_entries), 1):
        for j in range(len(vec_min_entry)):
            if vec_min_entry[j] > mat_entries[i][j]:
                vec_min_entry[j] = mat_entries[i][j]
            if vec_max_entry[j] < mat_entries[i][j]:
                vec_min_entry[j] = mat_entries[i][j]

    for i in range(var_count_clusters):
        vec_cluster_center = []
        for j in range(len(vec_min_entry)):
            vec_cluster_center.append(random.uniform(vec_min_entry[j], vec_max_entry[j]))
        mat_cluster_centers.append(vec_cluster_center)

    return mat_cluster_centers


def random_cluster_centers_from_entries(mat_entries, var_count_clusters):
    """
    :param mat_entries: vector of entry vectors
    :param var_count_clusters: - count clusters
    :return: vector of cluster centers vectors
    """

    mat_cluster_centers = []
    mat_cluster_centers_indexes = []
    for i in range(var_count_clusters):
        var_entry_inx = random.randint(0, len(mat_entries) - 1)
        while var_entry_inx in mat_cluster_centers_indexes:
            var_entry_inx = random.randint(0, len(mat_entries) - 1)
        mat_cluster_centers_indexes.append(var_entry_inx)
        mat_cluster_centers.append(mat_entries[var_entry_inx][:])
    return mat_cluster_centers


def k_means_centers_moving(mat_cluster_centers, mat_entries, mat_cluster_entry_indexes):
    """
    :param mat_cluster_centers: vector of cluster centers vectors
    :param mat_entries: vector of entry vectors
    :param mat_cluster_entry_indexes: vector of vectors with indexes
           of entries (in mat_entries) for entries of each cluster
    :return: vector of cluster centers vectors
    """

    for cl_num in range(len(mat_cluster_centers)):
        for i in range(len(mat_cluster_centers[0])):
            mat_cluster_centers[cl_num][i] = 0.0

        for en_num in mat_cluster_entry_indexes[cl_num]:
            for i in range(len(mat_entries[0])):
                mat_cluster_centers[cl_num][i] += mat_entries[en_num][i]

        var_ce_count = float(len(mat_cluster_entry_indexes[cl_num]))
        for i in range(len(mat_cluster_centers[0])):
            mat_cluster_centers[cl_num][i] /= var_ce_count
    return mat_cluster_centers


def c_means_centers_moving(mat_cluster_centers, mat_entries, mat_membership, mat_cluster_entry_indexes):
    """
    :param mat_cluster_centers: vector of cluster centers vectors
    :param mat_entries: vector of entry vectors
    :param mat_membership: vector of vectors with membership values for each cluster
    :param mat_cluster_entry_indexes: vector of vectors with indexes
           of entries (in mat_entries) for entries of each cluster
    :return: vector of cluster centers vectors
    """

    for cl in range(len(mat_cluster_centers)):
        var_membership_sum = 0.000001
        for en_num in mat_cluster_entry_indexes[cl]:
            var_membership_sum += mat_membership[cl][en_num]
        for i in range(len(mat_cluster_centers[cl])):
            mat_cluster_centers[cl][i] = 0.0
        for en_num in mat_cluster_entry_indexes[cl]:
            for i in range(len(mat_cluster_centers[cl])):
                mat_cluster_centers[cl][i] += mat_entries[en_num][i] * mat_membership[cl][en_num]
        for i in range(len(mat_cluster_centers[cl])):
            mat_cluster_centers[cl][i] /= var_membership_sum
    return mat_cluster_centers

