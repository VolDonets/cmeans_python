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


def mahalanobis_membership_matrix(mat_entries, mat_cluster_centers, ten_covariances):
    mat_membership = []
    for i in range(len(mat_cluster_centers)):
        vec_cluster_distances = []
        for j in range(len(mat_entries)):
            var_distance = distances.mahalanobis_distance(mat_entries[j],
                                                          mat_cluster_centers[i], ten_covariances[i])
            vec_cluster_distances.append(var_distance)
        mat_membership.append(vec_cluster_distances)
    return mat_membership


def mahalanobis_cauchy_membership_matrix(mat_entries, mat_cluster_centers, ten_covariances):
    mat_membership = []
    for i in range(len(mat_cluster_centers)):
        vec_cluster_distances = []
        for j in range(len(mat_entries)):
            var_distance = distances.mahalanobis_distance(mat_entries[j],
                                                          mat_cluster_centers[i], ten_covariances[i])
            var_distance = distributions.cauchy_distribution(1, var_distance)
            vec_cluster_distances.append(var_distance)
        mat_membership.append(vec_cluster_distances)
    return mat_membership


def mahalanobis_own_old_distribution(mat_entries, mat_cluster_centers, ten_covariances):
    mat_membership = []
    for i in range(len(mat_cluster_centers)):
        vec_cluster_distances = []
        for j in range(len(mat_entries)):
            var_distance = distances.mahalanobis_distance(mat_entries[j],
                                                          mat_cluster_centers[i], ten_covariances[i])
            var_distance = distributions.own_old_distribution(1, 0.5, var_distance)
            vec_cluster_distances.append(var_distance)
        mat_membership.append(vec_cluster_distances)
    return mat_membership


def norm_membership_matrix_for_clusters(mat_membership):
    vec_summarized_parameters = []
    mat_norm_membership = []
    for j in range(len(mat_membership)):
        var_sum_param = 0.0
        for i in range(len(mat_membership[0])):
            var_sum_param += mat_membership[j][i]
        var_sum_param /= len(mat_membership[0])
        vec_summarized_parameters.append(var_sum_param)
    for i in range(len(mat_membership[0])):
        vec_norm_membership = []
        for j in range(len(mat_membership)):
            var_norm_membership = mat_membership[j][i] / vec_summarized_parameters[j]
            vec_norm_membership.append(var_norm_membership)
        mat_norm_membership.append(vec_norm_membership)
    return mat_norm_membership


def norm_membership_matrix_for_entries(mat_membership):
    vec_summarized_parameters = []
    mat_norm_membership = []
    for j in range(len(mat_membership[0])):
        var_sum_param = 0.0
        for i in range(len(mat_membership)):
            var_sum_param += mat_membership[i][j]
        var_sum_param /= len(mat_membership)
        vec_summarized_parameters.append(var_sum_param)
    for i in range(len(mat_membership[0])):
        vec_norm_membership = []
        for j in range(len(mat_membership)):
            var_norm_membership = mat_membership[j][i] / vec_summarized_parameters[j]
            vec_norm_membership.append(var_norm_membership)
        mat_norm_membership.append(vec_norm_membership)
    return mat_norm_membership
