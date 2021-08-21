import distances
import distributions


def euclid_membership_matrix(mat_entries, mat_cluster_centers):
    mat_membership = []
    for vec_cluster_center in mat_cluster_centers:
        vec_cluster_distances = []
        for vec_x in mat_entries:
            var_distance = distances.euclidean_distance(vec_x, vec_cluster_center)
            vec_cluster_distances.append(var_distance)
        mat_membership.append(vec_cluster_distances)
    return mat_membership


def manhattan_membership_matrix(mat_entries, mat_cluster_centers):
    mat_membership = []
    for vec_cluster_center in mat_cluster_centers:
        vec_cluster_distances = []
        for vec_x in mat_entries:
            var_distance = distances.manhattan_distance(vec_x, vec_cluster_center)
            vec_cluster_distances.append(var_distance)
        mat_membership.append(vec_cluster_distances)
    return mat_membership


def mahalanobis_membership_matrix(mat_entries, mat_cluster_centers, ten_cluster_entries):
    mat_membership = []
    for i in range(len(mat_cluster_centers)):
        vec_cluster_distances = []
        for j in range(len(mat_entries)):
            var_distance = distances.mahalanobis_distance(mat_entries[j], mat_cluster_centers[i], ten_cluster_entries[i])
            vec_cluster_distances.append(var_distance)
        mat_membership.append(vec_cluster_distances)
    return mat_membership


def mahalanobis_cauchy_membership_matrix(mat_entries, mat_cluster_centers, ten_cluster_entries):
    mat_membership = []
    for i in range(len(mat_cluster_centers)):
        vec_cluster_distances = []
        for j in range(len(mat_entries)):
            var_distance = distances.mahalanobis_distance(mat_entries[j], mat_cluster_centers[i], ten_cluster_entries[i])
            var_distance = distributions.cauchy_distribution(1, var_distance)
            vec_cluster_distances.append(var_distance)
        mat_membership.append(vec_cluster_distances)
    return mat_membership

