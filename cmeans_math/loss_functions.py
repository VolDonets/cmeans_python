"""Loss functions

This script contains some loss functions

Functions:
    * simple_manhattan_loss
    * manhattan_loss
    * simple_mahalanobis_loss
    * mahalanobis_loss
    * inverse_mahalanobis_loss
    * divergence_loss
"""

import cluster_loss_functions


def simple_manhattan_loss(mat_entries, mat_cluster_centers):
    return sum(cluster_loss_functions.simple_manhattan_clusters_loss(mat_entries, mat_cluster_centers))


def manhattan_loss(mat_membership, mat_entries, mat_cluster_centers):
    return sum(cluster_loss_functions.manhattan_clusters_loss(mat_membership, mat_entries, mat_cluster_centers))


def simple_mahalanobis_loss(mat_entries, mat_cluster_centers, ten_covariances):
    return sum(cluster_loss_functions.simple_mahalanobis_clusters_loss(mat_entries,
                                                                       mat_cluster_centers, ten_covariances))


def mahalanobis_loss(mat_membership, mat_entries, mat_cluster_centers, ten_covariances):
    return sum(cluster_loss_functions.mahalanobis_clusters_loss(mat_membership, mat_entries, mat_cluster_centers))


def inverse_mahalanobis_loss(mat_membership, mat_entries, mat_cluster_centers, ten_covariances):
    return sum(cluster_loss_functions.inverse_mahalanobis_clusters_loss(mat_membership, mat_entries,
                                                                        mat_cluster_centers, ten_covariances))


def divergence_loss(mat_divergences):
    return sum(cluster_loss_functions.divergence_clusters_loss(mat_divergences))
