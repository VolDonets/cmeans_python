import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from numpy import interp
from itertools import cycle


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


def draw_roc_curve(mat_cluster_entry_indexes, vec_correct_entry_class, var_count_cluster):
    # to correct mat
    mat_correct_clusters = []
    for i in range(len(vec_correct_entry_class)):
        vec_correct_cluster = [0] * var_count_cluster
        vec_correct_cluster[vec_correct_entry_class[i]] = 1
        mat_correct_clusters.append(vec_correct_cluster)

    # to answered mat
    mat_answered_clusters = []
    for i in range(len(vec_correct_entry_class)):
        vec_answered_cluster = [0.0] * var_count_cluster
        mat_answered_clusters.append(vec_answered_cluster)

    for i in range(var_count_cluster):
        for j in range(len(mat_cluster_entry_indexes[i])):
            mat_answered_clusters[mat_cluster_entry_indexes[i][j]][i] = 1.0

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    print(mat_correct_clusters)
    print(mat_answered_clusters)

    mat_correct_clusters = np.array(mat_correct_clusters)
    mat_answered_clusters = np.array(mat_answered_clusters)

    for i in range(var_count_cluster):
        fpr[i], tpr[i], _ = roc_curve(mat_correct_clusters[:, i], mat_answered_clusters[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(mat_correct_clusters.ravel(), mat_answered_clusters.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure()
    lw = 2
    plt.plot(fpr[2], tpr[2], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.show()


def draw_multilabel_roc_curve(mat_cluster_entry_indexes, vec_correct_entry_class, var_count_cluster):
    # to correct mat
    mat_correct_clusters = []
    for i in range(len(vec_correct_entry_class)):
        vec_correct_cluster = [0] * var_count_cluster
        vec_correct_cluster[vec_correct_entry_class[i]] = 1
        mat_correct_clusters.append(vec_correct_cluster)

    # to answered mat
    mat_answered_clusters = []
    for i in range(len(vec_correct_entry_class)):
        vec_answered_cluster = [0] * var_count_cluster
        mat_answered_clusters.append(vec_answered_cluster)

    for i in range(var_count_cluster):
        for j in range(len(mat_cluster_entry_indexes[i])):
            mat_answered_clusters[mat_cluster_entry_indexes[i][j]][i] = 1

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    print(mat_correct_clusters)
    print(mat_answered_clusters)

    mat_correct_clusters = np.array(mat_correct_clusters)
    mat_answered_clusters = np.array(mat_answered_clusters)

    for i in range(var_count_cluster):
        fpr[i], tpr[i], _ = roc_curve(mat_correct_clusters[:, i], mat_answered_clusters[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(mat_correct_clusters.ravel(), mat_answered_clusters.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(var_count_cluster)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(var_count_cluster):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= var_count_cluster

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    # plt.plot(fpr["micro"], tpr["micro"],
    #          label='micro-average ROC curve (area = {0:0.2f})'
    #                ''.format(roc_auc["micro"]),
    #          color='deeppink', linestyle=':', linewidth=4)
    #
    # plt.plot(fpr["macro"], tpr["macro"],
    #          label='macro-average ROC curve (area = {0:0.2f})'
    #                ''.format(roc_auc["macro"]),
    #          color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green'])
    lw = 2
    for i, color in zip(range(var_count_cluster), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for each class')
    plt.legend(loc="lower right")
    plt.show()
