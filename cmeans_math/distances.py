from scipy.spatial import distance


def euclidean_distance(vec_x1, vec_x2):
    return distance.euclidean(vec_x1, vec_x2)


def manhattan_distance(vec_x1, vec_x2):
    return distance.cityblock(vec_x1, vec_x2)


def mahalanobis_distance(vec_x1, vec_x2, mat_E):
    return distance.mahalanobis(vec_x1, vec_x2, mat_E)
