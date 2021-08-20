from scipy.spatial import distance


SMALL_VALUE = 0.000001


def euclidean_distance(vector_x1, vector_x2):
    return distance.euclidean(vector_x1, vector_x2)


def manhattan_distance(vector_x1, vector_x2):
    return distance.cityblock(vector_x1, vector_x2)


def mahalanobis_distance(vector_x1, vector_x2, matrix_E):
    return distance.mahalanobis(vector_x1, vector_x2, matrix_E)
