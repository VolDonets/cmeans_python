import math


def cauchy_distribution(var_etta, var_distance):
    var_tmp = math.pi * var_etta * (1 + (var_distance / (var_etta * var_etta)))
    return 1 / var_tmp
