import numpy as np


class BasicDataSaver:
    def __init__(self, vec_param_names, mat_data, vec_check):
        self.vec_param_names = vec_param_names
        self.mat_entries = mat_data
        self.vec_check = vec_check


def load_iris_data():
    data = np.loadtxt("test_data/iris_data/names.txt", dtype='str', delimiter='|')
    vec_names = data.tolist()
    vec_names.pop(0)

    data = np.loadtxt("test_data/iris_data/data.txt", dtype='float', delimiter='|')
    mat_data = data.tolist()

    for vec_data in mat_data:
        vec_data.pop(0)

    data = np.loadtxt("test_data/iris_data/check.txt", dtype='str', delimiter='|')
    vec_name_check = data.tolist()

    vec_check = []
    vec_check_only_names = []
    for x in vec_name_check:
        if x not in vec_check_only_names:
            vec_check_only_names.append(x)
    for x in vec_name_check:
        vec_check.append(vec_check_only_names.index(x))

    return BasicDataSaver(vec_names, mat_data, vec_check)
