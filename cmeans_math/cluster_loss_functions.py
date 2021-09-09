"""Loss functions

This script contains some loss functions, which calculate loss for each cluster

Functions:
    * simple_manhattan_clusters_loss
    * manhattan_clusters_loss
    * simple_mahalanobis_clusters_loss
    * mahalanobis_clusters_loss
    * inverse_mahalanobis_clusters_loss
    * divergence_clusters_loss
"""

import distances


def simple_manhattan_clusters_loss(mat_entries, mat_cluster_centers):
    vec_clusters_loss = []
    for i in range(len(mat_cluster_centers)):
        var_loss = 0.0
        for j in range(len(mat_entries)):
            var_loss += distances.manhattan_distance(mat_cluster_centers[i], mat_entries[j])
        vec_clusters_loss.append(var_loss)
    return vec_clusters_loss


def manhattan_clusters_loss(mat_membership, mat_entries, mat_cluster_centers):
    vec_clusters_loss = []
    for i in range(len(mat_cluster_centers)):
        var_loss = 0.0
        for j in range(len(mat_entries)):
            var_loss += mat_membership[i][j] * distances.manhattan_distance(mat_cluster_centers[i], mat_entries[j])
        vec_clusters_loss.append(var_loss)
    return vec_clusters_loss


def simple_mahalanobis_clusters_loss(mat_entries, mat_cluster_centers, ten_covariances):
    vec_clusters_loss = []
    for i in range(len(mat_cluster_centers)):
        var_loss = 0.0
        for j in range(len(mat_entries)):
            var_loss += distances.mahalanobis_distance(mat_entries[j], mat_cluster_centers[i], ten_covariances[i])
        vec_clusters_loss.append(var_loss)
    return vec_clusters_loss


def mahalanobis_clusters_loss(mat_membership, mat_entries, mat_cluster_centers, ten_covariances):
    vec_clusters_loss = []
    for i in range(len(mat_cluster_centers)):
        var_loss = 0.0
        for j in range(len(mat_entries)):
            var_loss += mat_membership[i][j] * \
                        distances.mahalanobis_distance(mat_entries[j], mat_cluster_centers[i], ten_covariances[i])
        vec_clusters_loss.append(var_loss)
    return vec_clusters_loss


def inverse_mahalanobis_clusters_loss(mat_membership, mat_entries, mat_cluster_centers, ten_covariances):
    vec_clusters_loss = []
    for i in range(len(mat_cluster_centers)):
        var_loss = 0.0
        for j in range(len(mat_entries)):
            var_loss += (1.0 / mat_membership[i][j]) * \
                        distances.mahalanobis_distance(mat_entries[j], mat_cluster_centers[i], ten_covariances[i])
        vec_clusters_loss.append(var_loss)
    return vec_clusters_loss


def divergence_clusters_loss(mat_divergences):
    vec_clusters_loss = []
    for i in range(len(mat_divergences)):
        var_loss = 0.0
        for j in range(len(mat_divergences[0])):
            var_loss -= mat_divergences[i][j]
        vec_clusters_loss.append(var_loss)
    return vec_clusters_loss
