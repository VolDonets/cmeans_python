"""Divergences (like information distance

Functions:
    * kulback_leibler_divergence
    * jensen_shannon_divergence
    * cross_entropy
"""

import math


SMALL_VALUE = 0.000001


def kulback_leibler_divergence(var_old_distance, var_new_distance):
    if var_old_distance == 0:
        var_old_distance = SMALL_VALUE
    return var_new_distance * math.log2(var_new_distance / var_old_distance)


def jensen_shannon_divergence(var_old_distance, var_new_distance):
    if var_old_distance == 0:
        var_old_distance = SMALL_VALUE
    return var_new_distance - var_old_distance * math.log2(var_new_distance / var_old_distance)


def cross_entropy(var_old_distance, var_new_distance):
    if var_old_distance == 0:
        var_old_distance = SMALL_VALUE
    return var_new_distance * math.log2(var_old_distance)


def divergence(mat_membership, mat_old_membership,
               divergence_func=kulback_leibler_divergence):
    mat_divergence = []
    for j in range(len(mat_membership)):
        vec_divergence = []
        for i in range(len(mat_membership[0])):
            vec_divergence.append(divergence_func(mat_old_membership[j][i],
                                                  mat_membership[j][i]))
        mat_divergence.append(vec_divergence)
    return mat_divergence
