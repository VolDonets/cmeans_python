"""Covariances

This script contains function for generating covariance matrix

Functions:
    * np_covariance_matrix: numpy calculating of covariance matrix
    * np_inverse_covariance: numpy calculating inverse matrix
    * cluster_covariance: calculate inverse and reguralized covariance matrix for clustering
"""

import numpy as np
from scipy import linalg


ARRAY_LAMBDA = 0.001


def np_covariance_matrix(mat_entries, vec_cluster_indexes, vec_cluster_center):
    """
    :param mat_entries: vector of entry vectors
    :param vec_cluster_indexes: vector of indexes selected entries
    :param vec_cluster_center: vector of cluster center coordinates
    :return: numpy covariance matrix
    """

    mat_cluster_entries = []
    if len(vec_cluster_indexes) != 0:
        for inx in vec_cluster_indexes:
            mat_cluster_entries.append(mat_entries[inx])
    else:
        mat_cluster_entries.append(vec_cluster_center)
        mat_cluster_entries.append(vec_cluster_center[:])
    if len(vec_cluster_indexes) == 1:
        mat_cluster_entries.append(vec_cluster_center)
    npmat_cluster_entries = np.array(mat_cluster_entries).T
    npmat_covariance = np.cov(npmat_cluster_entries)
    return npmat_covariance


def np_inverse_covariance(npmat_covariance):
    """
    :param npmat_covariance: numpy covariance matrix
    :return: numpy inverse covariance matrix
    """

    for i in range(len(npmat_covariance)):
        npmat_covariance[i][i] += ARRAY_LAMBDA
    return linalg.inv(npmat_covariance)


def cluster_covariances(mat_entries, mat_cluster_entry_indexes, mat_cluster_centers):
    """
    :param mat_entries: mat_entries: vector of entry vectors
    :param mat_cluster_entry_indexes: vector of vectors with indexes
           of entries (in mat_entries) for entries of each cluster
    :param mat_cluster_centers: vector of cluster centers vectors
    :return: prepared clustering inverse covariance matrix
    """

    ten_covariances = []
    for i in range(len(mat_cluster_entry_indexes)):
        npmat_covariance = np_covariance_matrix(mat_entries, mat_cluster_entry_indexes[i], mat_cluster_centers[i])
        npmat_covariance = np_inverse_covariance(npmat_covariance)
        ten_covariances.append(npmat_covariance.tolist())
    return ten_covariances
