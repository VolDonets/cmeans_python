import math


def cauchy_distribution(var_etta, var_distance):
    var_tmp = math.pi * var_etta * (1 + (var_distance / (var_etta * var_etta)))
    return 1 / var_tmp


def own_old_distribution(var_etta, var_betta, var_distance):
    var_pow = 1 / (1 - var_betta)
    var_tmp = var_distance / (var_etta * var_etta)
    var_tmp = math.pow(var_tmp, var_pow)
    var_tmp = var_tmp + 1
    var_tmp = 1 / var_tmp
    return var_tmp
