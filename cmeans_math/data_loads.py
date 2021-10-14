import numpy as np
import pandas as pd


class BasicDataSaver:
    def __init__(self, vec_param_names, mat_data, vec_check):
        self.vec_param_names = vec_param_names
        self.mat_entries = mat_data
        self.vec_check = vec_check


def load_iris_data(path=""):
    data = np.loadtxt(path + "test_data/iris_data/names.txt", dtype='str', delimiter='|')
    vec_names = data.tolist()
    vec_names.pop(0)

    data = np.loadtxt(path + "test_data/iris_data/data.txt", dtype='float', delimiter='|')
    mat_data = data.tolist()

    for vec_data in mat_data:
        vec_data.pop(0)

    data = np.loadtxt(path + "test_data/iris_data/check.txt", dtype='str', delimiter='|')
    vec_name_check = data.tolist()

    vec_check = []
    vec_check_only_names = []
    for x in vec_name_check:
        if x not in vec_check_only_names:
            vec_check_only_names.append(x)
    for x in vec_name_check:
        vec_check.append(vec_check_only_names.index(x))

    return BasicDataSaver(vec_names, mat_data, vec_check)


def load_urology_cleaned():
    data = np.loadtxt("test_data/medical_data_cleaned/names.txt", dtype='str', delimiter='|')
    vec_names = data.tolist()
    vec_names.pop(0)

    data = np.loadtxt("test_data/medical_data_cleaned/data.txt", dtype='float', delimiter='|')
    mat_data = data.tolist()

    for vec_data in mat_data:
        vec_data.pop(0)

    data = np.loadtxt("test_data/medical_data_cleaned/check.txt", dtype='str', delimiter='|')
    vec_name_check = data.tolist()

    vec_check = []
    vec_check_only_names = []
    for x in vec_name_check:
        if x not in vec_check_only_names:
            vec_check_only_names.append(x)
    for x in vec_name_check:
        vec_check.append(vec_check_only_names.index(x))

    return BasicDataSaver(vec_names, mat_data, vec_check)


def load_from_csv(path_to_csv_file, class_name, vec_drop_column=[]):
    csv_data = pd.read_csv(path_to_csv_file)
    if len(vec_drop_column) > 0:
        csv_data = csv_data.drop(columns=vec_drop_column)

    vec_param_names = csv_data.columns.tolist()[:]
    vec_param_names.remove(class_name)

    vec_check = csv_data[class_name].tolist()

    csv_data = csv_data.drop(columns=[class_name])
    mat_entries = []
    for i in range(csv_data.shape[0]):
        mat_entries.append(csv_data.iloc[i, :].tolist())

    return BasicDataSaver(vec_param_names, mat_entries, vec_check)


if __name__ == "__main__":
    data = load_from_csv('../test_data/urology_prepared/original_clear.csv', 'cluster', ['Unnamed: 0'])
    print(data.vec_check)
    print(data.vec_param_names)
    print(data.mat_entries)
