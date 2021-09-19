"""Loss functions

This script contains some loss functions

Functions:
    * manhattan_loss: calculates loss value by sum of Manhattan distance mul membership value
    * euclid_loss: calculates loss value by sum of Euclid distance mul membership value
    * mahalanobis_loss: calculates loss value by sum of Mahalanobis distance mul membership value
    * inverse_mahalanobis_loss: calculates loss value by sum of Mahalanobis distance mul inverse membership value
    * divergence_loss: calculates loss value by sum of manhattan distance mul membership value
"""

import distances


def manhattan_loss(mat_membership, mat_entries, mat_cluster_centers):
    """
    manhattan_loss = ∑j ∑i w_ji * d_manhattan^2(entry_i, center_j)

    :param mat_membership: vector of vector membership values for the each cluster by entry
    :param mat_entries: vector of entry vectors
    :param mat_cluster_centers: vector of cluster centers vectors
    :return: manhattan_loss
    """

    var_loss = 0.0
    for j in range(len(mat_cluster_centers)):
        for i in range(len(mat_entries)):
            var_distance = distances.manhattan_distance(mat_cluster_centers[j], mat_entries[i]) ** 2
            var_loss += mat_membership[j][i] * var_distance
    return var_loss


def euclid_loss(mat_membership, mat_entries, mat_cluster_centers):
    """
    euclid_loss = ∑j ∑i w_ji * d_euclid^2(entry_i, center_j)

    :param mat_membership: vector of vector membership values for the each cluster by entry
    :param mat_entries: vector of entry vectors
    :param mat_cluster_centers: vector of cluster centers vectors
    :return: euclid_loss
    """

    var_loss = 0.0
    for j in range(len(mat_cluster_centers)):
        for i in range(len(mat_entries)):
            var_distance = distances.euclidean_distance(mat_cluster_centers[j], mat_entries[i]) ** 2
            var_loss += mat_membership[j][i] * var_distance
    return var_loss


def mahalanobis_loss(mat_membership, mat_entries, mat_cluster_centers, ten_covariances):
    """
    mahalanobis_loss = ∑j ∑i w_ji * d_mahalanobis^2(center_j, entry_i, covariance_j)

    :param mat_membership: vector of vector membership values for the each cluster by entry
    :param mat_entries: vector of entry vectors
    :param mat_cluster_centers: vector of cluster centers vectors
    :param ten_covariances: vector of covariances matrices
    :return: mahalanobis_loss
    """

    var_loss = 0.0
    for j in range(len(mat_cluster_centers)):
        for i in range(len(mat_entries)):
            var_distance = distances.mahalanobis_distance(mat_cluster_centers[j], mat_entries[i], ten_covariances[j]) ** 2
            var_loss += mat_membership[j][i] * var_distance
    return var_loss


def inverse_mahalanobis_loss(mat_membership, mat_entries, mat_cluster_centers, ten_covariances):
    """
    inverse_mahalanobis_loss = ∑j ∑i w_ji^(-1) * d_mahalanobis^2(center_j, entry_i, covariance_j)

    :param mat_membership: vector of vector membership values for the each cluster by entry
    :param mat_entries: vector of entry vectors
    :param mat_cluster_centers: vector of cluster centers vectors
    :param ten_covariances: vector of covariances matrices
    :return: inverse_mahalanobis_loss
    """

    var_loss = 0.0
    for j in range(len(mat_cluster_centers)):
        for i in range(len(mat_entries)):
            var_distance = distances.mahalanobis_distance(mat_cluster_centers[j], mat_entries[i],
                                                          ten_covariances[j]) ** 2
            var_loss += var_distance / mat_membership[j][i]
    return var_loss


def divergence_loss(mat_divergences):
    """
    divergence_loss = ∑j ∑i divergence_ji
    :param mat_divergences:
    :return:
    """
    var_loss = 0.0
    for j in range(len(mat_divergences)):
        for i in range(len(mat_divergences[0])):
            var_loss -= mat_divergences[j][i]
    return var_loss
