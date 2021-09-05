import numpy as np
from scipy import linalg


ARRAY_LAMBDA = 0.001


def np_covariance_matrix(mat_entries, vec_cluster_indexes):
    mat_cluster_entries = []
    for inx in vec_cluster_indexes:
        mat_cluster_entries.append(mat_entries[inx])
    npmat_cluster_entries = np.array(mat_cluster_entries).T
    npmat_covariance = np.cov(npmat_cluster_entries)
    return npmat_covariance


def np_inverse_covariance(npmat_covariance):
    for i in range(len(npmat_covariance)):
        npmat_covariance[i][i] += ARRAY_LAMBDA
    return linalg.inv(npmat_covariance)


def cluster_covariances(mat_entries, mat_cluster_entry_indexes):
    ten_covariances = []
    for i in range(len(mat_cluster_entry_indexes)):
        npmat_covariance = np_covariance_matrix(mat_entries, mat_cluster_entry_indexes[i])
        npmat_covariance = np_inverse_covariance(npmat_covariance)
        ten_covariances.append(npmat_covariance.tolist())
    return ten_covariances
