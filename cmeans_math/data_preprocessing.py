import numpy as np


def get_min_vec_from_data_set(mat_entries):
    vec_min = []
    for x in mat_entries[0]:
        vec_min.append(x)

    for vec_entry in mat_entries:
        for inx in range(len(vec_entry)):
            if vec_entry[inx] < vec_min[inx]:
                vec_min[inx] = vec_entry[inx]

    return vec_min


def get_max_vec_from_data_set(mat_entries):
    vec_max = []
    for x in mat_entries[0]:
        vec_max.append(x)

    for vec_entry in mat_entries:
        for inx in range(len(vec_entry)):
            if vec_entry[inx] > vec_max[inx]:
                vec_max[inx] = vec_entry[inx]

    return vec_max


def get_updated_data_set(mat_entries, var_min_border=-1.0, var_max_border=1.0):
    mat_entries_up = []
    vec_min = get_min_vec_from_data_set(mat_entries)
    vec_max = get_max_vec_from_data_set(mat_entries)

    for vec_entry in mat_entries:
        vec_entry_up = []
        for inx in range(len(vec_entry)):
            vec_entry_up.append((vec_entry[inx] - vec_min[inx]) / (vec_max[inx] - vec_min[inx]))
            vec_entry_up[inx] = (vec_entry_up[inx] * (var_max_border - var_min_border)) + var_min_border
        mat_entries_up.append(vec_entry_up)

    return mat_entries_up


def get_norm_entries(mat_entries):
    np_mat_entries = np.array(mat_entries)
    np_vec_mean = np_mat_entries.mean(axis=0)
    np_vec_std = np_mat_entries.std(axis=0)

    np_mat_entries_norm = np_mat_entries - np_vec_mean
    np_mat_entries_norm = np_mat_entries_norm / np_vec_std

    mat_entries_norm = np_mat_entries_norm.tolist()
    return mat_entries_norm
