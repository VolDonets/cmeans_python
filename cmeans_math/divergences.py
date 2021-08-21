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
