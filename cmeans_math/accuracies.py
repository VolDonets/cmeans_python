
def accuracy(mat_cluster_entry_indexes, vec_correct_entry_class):
    var_accuracy = 0.0
    for cl_inx in range(len(mat_cluster_entry_indexes)):
        for en_inx in range(len(mat_cluster_entry_indexes[cl_inx])):
            if vec_correct_entry_class[mat_cluster_entry_indexes[cl_inx][en_inx]] == cl_inx:
                var_accuracy += 1.0
    var_accuracy /= len(vec_correct_entry_class)
    return var_accuracy


def confusion_matrix(mat_cluster_entry_indexes, vec_correct_entry_class):
    mat_confusion = []
    for i in range(len(mat_cluster_entry_indexes)):
        vec_confusion = []
        for j in range(len(mat_cluster_entry_indexes)):
            vec_confusion.append(0)
        mat_confusion.append(vec_confusion)

    for cl in range(len(mat_cluster_entry_indexes)):
        for el in range(len(mat_cluster_entry_indexes[cl])):
            mat_confusion[vec_correct_entry_class[mat_cluster_entry_indexes[cl][el]]][cl] += 1

    return mat_confusion
