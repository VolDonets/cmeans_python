"""Loss functions

This script contains some density functions, which calculate density for each cluster

Functions:
    * manhattan_density: returns densities for each cluster center
    * euclid_density: -//-
    * simple_mahalanobis_density: -//-
    * mahalanobis_density: -//-
    * inverse_mahalanobis_density: -//-
    * divergence_density: -//-
"""
import math

import distances
import memberships


BIG_NUMBER = 0.0


def simple_manhattan_density(mat_entries, mat_cluster_centers, mat_cluster_entry_indexes):
    """
    manhattan_density_j = 1/(count i) ∑i d_manhattan^2(entry_i, center_j)

    :param mat_entries: vector of entry vectors
    :param mat_cluster_centers: vector of cluster centers vectors
    :param mat_cluster_entry_indexes: vector of vectors with cluster entry indexes
    :return: vector of manhattan densities
    """

    vec_densities = []
    for j in range(len(mat_cluster_centers)):
        var_density = 0.0
        if len(mat_cluster_entry_indexes[j]) > 0:
            for i in mat_cluster_entry_indexes[j]:
                var_distance = distances.manhattan_distance(mat_cluster_centers[j], mat_entries[i])
                var_density += var_distance
            var_density /= len(mat_cluster_entry_indexes[j])
        else:
            var_density = BIG_NUMBER
        vec_densities.append(var_density)
    return vec_densities


def manhattan_density(mat_membership, mat_entries, mat_cluster_centers, mat_cluster_entry_indexes):
    """
    manhattan_density_j = 1/(count i) ∑i w_ji * d_manhattan^2(entry_i, center_j)

    :param mat_membership: vector of vector membership values for the each cluster by entry
    :param mat_entries: vector of entry vectors
    :param mat_cluster_centers: vector of cluster centers vectors
    :param mat_cluster_entry_indexes: vector of vectors with cluster entry indexes
    :return: vector of manhattan densities
    """

    vec_densities = []
    for j in range(len(mat_cluster_centers)):
        var_density = 0.0
        if len(mat_cluster_entry_indexes[j]) > 0:
            for i in mat_cluster_entry_indexes[j]:
                var_distance = distances.manhattan_distance(mat_cluster_centers[j], mat_entries[i]) ** 2
                var_density += mat_membership[j][i] * var_distance
            var_density /= len(mat_cluster_entry_indexes[j])
        else:
            var_density = BIG_NUMBER
        vec_densities.append(var_density)
    return vec_densities


def euclid_density(mat_membership, mat_entries, mat_cluster_centers, mat_cluster_entry_indexes):
    """
    euclid_density_j = 1/(count i) ∑i w_ji * d_euclid^2(entry_i, center_j)

    :param mat_membership: vector of vector membership values for the each cluster by entry
    :param mat_entries: vector of entry vectors
    :param mat_cluster_centers: vector of cluster centers vectors
    :param mat_cluster_entry_indexes: vector of vectors with cluster entry indexes
    :return: vector of euclid densities
    """

    vec_densities = []
    for j in range(len(mat_cluster_centers)):
        var_density = 0.0
        if len(mat_cluster_entry_indexes[j]) > 0:
            for i in mat_cluster_entry_indexes[j]:
                var_distance = distances.euclidean_distance(mat_cluster_centers[j], mat_entries[i]) ** 2
                var_density += mat_membership[j][i] * var_distance
            var_density /= len(mat_cluster_entry_indexes[j])
        else:
            var_density = BIG_NUMBER
        vec_densities.append(var_density)
    return vec_densities


def simple_mahalanobis_density(mat_entries, mat_cluster_centers, ten_covariances, mat_cluster_entry_indexes):
    """
    simple_mahalanobis_density_j = 1/(count i) ∑i MD^2(center_j, entry_i, covariance_j)

    :param mat_entries: vector of entry vectors
    :param mat_cluster_centers: vector of cluster centers vectors
    :param ten_covariances: vector of covariances matrices
    :param mat_cluster_entry_indexes: vector of vectors with cluster entry indexes
    :return: vector of simple mahalanobis densities
    """

    vec_densities = []
    for j in range(len(mat_cluster_centers)):
        var_density = 0.0
        if len(mat_cluster_entry_indexes[j]) > 0:
            for i in mat_cluster_entry_indexes[j]:
                var_distance = distances.mahalanobis_distance(mat_cluster_centers[j],
                                                              mat_entries[i], ten_covariances[j]) ** 2
                var_density += var_distance
            var_density /= len(mat_cluster_entry_indexes[j])
        else:
            var_density = BIG_NUMBER
        vec_densities.append(var_density)
    return vec_densities


def mahalanobis_density(mat_membership, mat_entries, mat_cluster_centers, ten_covariances, mat_cluster_entry_indexes):
    """
    mahalanobis_density_j = 1/(count i) ∑i w_ji * MD^2(center_j, entry_i, covariance_j)

    :param mat_membership: vector of vector membership values for the each cluster by entry
    :param mat_entries: vector of entry vectors
    :param mat_cluster_centers: vector of cluster centers vectors
    :param ten_covariances: vector of covariances matrices
    :param mat_cluster_entry_indexes: vector of vectors with cluster entry indexes
    :return: vector of mahalanobis densities
    """

    vec_densities = []
    for j in range(len(mat_cluster_centers)):
        var_density = 0.0
        if len(mat_cluster_entry_indexes[j]) > 0:
            for i in mat_cluster_entry_indexes[j]:
                var_distance = distances.mahalanobis_distance(mat_cluster_centers[j],
                                                              mat_entries[i], ten_covariances[j]) ** 2
                var_density += mat_membership[j][i] * var_distance
            var_density /= len(mat_cluster_entry_indexes[j])
        else:
            var_density = BIG_NUMBER
        vec_densities.append(var_density)
    return vec_densities


def inverse_mahalanobis_density(mat_membership, mat_entries, mat_cluster_centers,
                                ten_covariances, mat_cluster_entry_indexes):
    """
    inverse_mahalanobis_density_j = 1/(count i) ∑i w_ji^(-1) * MD^2(center_j, entry_i, covariance_j)

    :param mat_membership: vector of vector membership values for the each cluster by entry
    :param mat_entries: vector of entry vectors
    :param mat_cluster_centers: vector of cluster centers vectors
    :param ten_covariances: vector of covariances matrices
    :param mat_cluster_entry_indexes: vector of vectors with cluster entry indexes
    :return: vector of inverse mahalanobis densities
    """

    vec_densities = []
    for j in range(len(mat_cluster_centers)):
        var_density = 0.0
        if len(mat_cluster_entry_indexes[j]) > 0:
            for i in mat_cluster_entry_indexes[j]:
                var_distance = distances.mahalanobis_distance(mat_cluster_centers[j],
                                                              mat_entries[i], ten_covariances[j]) ** 2
                var_density += var_distance / (mat_membership[j][i] + 0.000001)
            var_density /= len(mat_cluster_entry_indexes[j])
        else:
            var_density = BIG_NUMBER
        vec_densities.append(var_density)
    return vec_densities


def inverse_manhattan_density(mat_membership, mat_entries, mat_cluster_centers, mat_cluster_entry_indexes):
    """
    inverse_manhattan_density_j = 1/(count i) ∑i w_ji^(-1) * d_1(center_j, entry_i)

    :param mat_membership: vector of vector membership values for the each cluster by entry
    :param mat_entries: vector of entry vectors
    :param mat_cluster_centers: vector of cluster centers vectors
    :param ten_covariances: vector of covariances matrices
    :param mat_cluster_entry_indexes: vector of vectors with cluster entry indexes
    :return: vector of inverse mahalanobis densities
    """

    vec_densities = []
    for j in range(len(mat_cluster_centers)):
        var_density = 0.0
        if len(mat_cluster_entry_indexes[j]) > 0:
            for i in mat_cluster_entry_indexes[j]:
                var_distance = distances.manhattan_distance(mat_cluster_centers[j],
                                                            mat_entries[i])
                var_density += var_distance / (mat_membership[j][i] + 0.000001)
            var_density /= len(mat_cluster_entry_indexes[j])
        else:
            var_density = BIG_NUMBER
        vec_densities.append(var_density)
    return vec_densities


def new_divergence_density(mat_entries, mat_cluster_centers,
                           ten_covariances, mat_cluster_entry_indexes,
                           get_divergence_membership_matrix_func=memberships.kulback_leibler_membership_matrix,
                           basic_distance="Manhattan"):
    vec_densities = []
    mat_kl_membership = get_divergence_membership_matrix_func(mat_entries=mat_entries,
                                                              mat_cluster_centers=mat_cluster_centers,
                                                              ten_covariances=ten_covariances,
                                                              basic_distance=basic_distance)
    for j in range(len(mat_cluster_centers)):
        var_density = 0.0
        if len(mat_cluster_entry_indexes[j]) > 0:
            for i in mat_cluster_entry_indexes[j]:
                var_minus_fix = -1.0 if mat_kl_membership[j][i] > 0 else 1.0
                var_density += mat_kl_membership[j][i] * var_minus_fix
            var_density /= len(mat_cluster_entry_indexes[j])
        else:
            var_density = BIG_NUMBER
        vec_densities.append(var_density)
    return vec_densities


def divergence_density(mat_divergences):
    """
    divergence_density_j = ∑j ∑i divergence_ji

    :param mat_divergences:
    :return: vector of divergence densities
    """

    vec_densities = []
    for j in range(len(mat_divergences)):
        var_density = 0.0
        for i in range(len(mat_divergences[0])):
            var_density -= mat_divergences[j][i]
        vec_densities.append(var_density)
    return vec_densities
