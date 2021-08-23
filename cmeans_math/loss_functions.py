import distances


def simple_manhattan_loss(mat_entries, mat_cluster_centers):
    var_loss = 0.0
    for i in range(len(mat_cluster_centers)):
        for j in range(len(mat_entries)):
            var_loss += distances.manhattan_distance(mat_cluster_centers[i], mat_entries[j])
    return var_loss


def manhattan_loss(mat_membership, mat_entries, mat_cluster_centers):
    var_loss = 0.0
    for i in range(len(mat_cluster_centers)):
        for j in range(len(mat_entries)):
            var_loss += mat_membership[i][j] * distances.manhattan_distance(mat_cluster_centers[i], mat_entries[j])
    return var_loss


def simple_mahalanobis_loss(mat_entries, mat_cluster_centers, ten_covariances):
    var_loss = 0.0
    for i in range(len(mat_cluster_centers)):
        for j in range(len(mat_entries)):
            var_loss += distances.mahalanobis_distance(mat_entries[j], mat_cluster_centers[i], ten_covariances[i])
    return var_loss


def mahalanobis_loss(mat_membership, mat_entries, mat_cluster_centers, ten_covariances):
    var_loss = 0.0
    for i in range(len(mat_cluster_centers)):
        for j in range(len(mat_entries)):
            var_loss += mat_membership[i][j] * \
                        distances.mahalanobis_distance(mat_entries[j], mat_cluster_centers[i], ten_covariances[i])
    return var_loss


def inverse_mahalanobis_loss(mat_membership, mat_entries, mat_cluster_centers, ten_covariances):
    var_loss = 0.0
    for i in range(len(mat_cluster_centers)):
        for j in range(len(mat_entries)):
            var_loss += (1.0 / mat_membership[i][j]) * \
                        distances.mahalanobis_distance(mat_entries[j], mat_cluster_centers[i], ten_covariances[i])
    return var_loss


def divergence_loss(mat_divergences):
    var_loss = 0.0
    for i in range(len(mat_divergences)):
        for j in range(len(mat_divergences[0])):
            var_loss -= mat_divergences[i][j]
    return var_loss
