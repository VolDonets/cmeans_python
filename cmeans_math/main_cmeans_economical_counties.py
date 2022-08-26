import cmeans_math.data_loads as data_loads
import cmeans_math.accuracies as accuracies
import cmeans_math.data_preprocessing as dp
import cmeans_math.clustering_results as cl_res

import matplotlib.pyplot as plt
from itertools import combinations

import numpy as np
from fcmeans import FCM


dataset = data_loads.load_from_csv('../test_data/economical_data/minmax_scaler_range01.csv', 'cluster', ['Country Name'])
# dataset = data_loads.load_from_csv('../test_data/economical_data/economical_pca_97pr.csv', 'cluster', ['no'])
# dataset = data_loads.load_from_csv('../test_data/economical_data/economical_pca_99pr.csv', 'cluster', ['no'])

X = np.array(dataset.mat_entries)
Y = np.array(dataset.vec_check)

par_coef = []
par_entr_coef = []
for i in range(1, 21):
    cmeans_tst = FCM(n_clusters=i, max_iter=300, random_state=0)
    cmeans_tst.fit(X)
    par_coef.append(cmeans_tst.partition_coefficient)
    par_entr_coef.append(cmeans_tst.partition_entropy_coefficient)

plt.plot(range(1, 21), par_coef)
plt.xlabel('Number of Clusters')
plt.ylabel('Partition Coefficient')
plt.show()

plt.plot(range(1, 21), par_entr_coef)
plt.xlabel('Number of Clusters')
plt.ylabel('Partition Entropy Coefficient')
plt.show()

cmeans = FCM(n_clusters=4, max_iter=300, random_state=1)
cmeans.fit(X)
pred_y = cmeans.predict(X)
plt.scatter(X[:,0], X[:,1])
plt.scatter(cmeans.centers[:, 0], cmeans.centers[:, 1], s=300, c='red')
plt.show()

mcei = []
for inx in range(4):
    mcei.append([])

pred_y = pred_y.tolist()

for inx in range(len(pred_y)):
    mcei[pred_y[inx]].append(inx)

mcei[0], mcei[2] = mcei[2], mcei[0]


obj_results = cl_res.ClusteringResults(mat_cluster_centers=cmeans.centers.tolist(),
                                       mat_cluster_entry_indexes=mcei)

obj_results.mat_cluster_centers[0], obj_results.mat_cluster_centers[2] = obj_results.mat_cluster_centers[2], \
                                                                         obj_results.mat_cluster_centers[0]

var_accuracy = accuracies.accuracy(obj_results.mat_cluster_entry_indexes, dataset.vec_check)
mat_confusion = accuracies.confusion_matrix(obj_results.mat_cluster_entry_indexes, dataset.vec_check)

print("accuracy:", var_accuracy)
print("Confusion")
for vec_confusion in mat_confusion:
    print(vec_confusion)

print(obj_results.mat_cluster_entry_indexes)
print(dataset.vec_check)

if len(obj_results.vec_cluster_count) > 0:
    x = obj_results.vec_cluster_count
    y = obj_results.vec_total_losses

    plt.plot(x, y)
    plt.xlabel('Clusters count')
    plt.ylabel('Total loss')
    plt.title('My cluster compactness')
    plt.show()
elif len(obj_results.vec_step_number) > 0:
    x = obj_results.vec_step_number
    y = obj_results.vec_total_losses

    plt.plot(x, y)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Loss changing')
    plt.show()

accuracies.draw_3d_clusters(obj_results.mat_cluster_centers, dataset.mat_entries,
                            obj_results.mat_cluster_entry_indexes,
                            vec_param_names=dataset.vec_param_names,
                            # vec_params=[0, 1, 10])
                            vec_params=[0, 1, 2])
                            # vec_params=[0, 1, 4])

accuracies.draw_3d_clusters(obj_results.mat_cluster_centers, dataset.mat_entries,
                            obj_results.mat_cluster_entry_indexes,
                            vec_param_names=dataset.vec_param_names,
                            # vec_params=[15, 17, 27])
                            vec_params=[1, 2, 3])
                            # vec_params=[7, 10, 11])

accuracies.draw_3d_clusters(obj_results.mat_cluster_centers, dataset.mat_entries,
                            obj_results.mat_cluster_entry_indexes,
                            vec_param_names=dataset.vec_param_names,
                            # vec_params=[13, 19, 30])
                            vec_params=[0, 2, 3])
                            # vec_params=[13, 15, 17])

accuracies.draw_roc_curve(obj_results.mat_cluster_entry_indexes,
                          dataset.vec_check,
                          3)

accuracies.draw_multilabel_roc_curve(obj_results.mat_cluster_entry_indexes,
                                     dataset.vec_check,
                                     3)

vec_colors = ['blue', 'red', 'pink', 'green', 'orange', 'lime', 'purple',
              'aqua', 'navy', 'coral', 'teal', 'mustard', 'black',
              'maroon', 'yellow']


def draw_clusters(pmat_cluster_centers, pten_cluster_entries, x_label_name, y_label_name, obj_ax, plot_number):
    obj_ax.subplot(2, 3, plot_number)
    for cl_inx in range(len(pmat_cluster_centers)):
        obj_ax.scatter(pmat_cluster_centers[cl_inx][0], pmat_cluster_centers[cl_inx][1], s=100, c=vec_colors[cl_inx])
        obj_ax.scatter(pten_cluster_entries[cl_inx][0], pten_cluster_entries[cl_inx][1], s=10, c=vec_colors[cl_inx],
                       marker='o', label='cl_' + str(cl_inx))
    obj_ax.legend(loc='upper left')
    obj_ax.xlabel(x_label_name)
    obj_ax.ylabel(y_label_name)
    obj_ax.tick_params(axis='both', which='major', labelsize=9)


def draw_clusters_single_dim(pmat_cluster_centers, pten_cluster_entries, x_label_name, y_label_name, title_name):
    for cl_inx in range(len(pmat_cluster_centers)):
        plt.scatter(pmat_cluster_centers[cl_inx][0], pmat_cluster_centers[cl_inx][1], s=100, c=vec_colors[cl_inx])
        plt.scatter(pten_cluster_entries[cl_inx][0], pten_cluster_entries[cl_inx][1], s=10, c=vec_colors[cl_inx])
    plt.title(title_name, fontsize=19)
    plt.xlabel(x_label_name, fontsize=10)
    plt.ylabel(y_label_name, fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=9)
    plt.show()


mat_entries = dataset.mat_entries
vec_param_names = dataset.vec_param_names
mat_cluster_centers = obj_results.mat_cluster_centers
mat_cluster_entry_indexes = obj_results.mat_cluster_entry_indexes

count_param = len(obj_results.mat_cluster_centers[0])
mat_axis_orig = list(combinations(range(count_param), 2))

for i in range(int(len(mat_axis_orig) / 6)):
    inx_from = i * 6
    inx_to = i * 6 + 6
    inx_to = inx_to if inx_to <= len(mat_axis_orig) else len(mat_axis_orig)
    # mat_axis = mat_axis_orig[inx_from:inx_to]
    # print('inx: ', inx_from, inx_to)

    # mat_axis = [mat_axis_orig[0],
    #             # mat_axis_orig[6],
    #             mat_axis_orig[20],
    #             mat_axis_orig[50],
    #             mat_axis_orig[69],
    #             mat_axis_orig[135],
    #             mat_axis_orig[170]]

    # interested params:
    # 0 3 10 15 17 27 30
    # mat_axis = [
    #     (0, 3),
    #     (4, 7),
    #     (10, 15),
    #     (17, 27),
    #     (19, 30),
    #     (12, 13)
    # ]

    mat_axis = [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3)
    ]

    # mat_axis = [
    #     (0, 4),
    #     (1, 2),
    #     (3, 4),
    #     (1, 3),
    #     (1, 4),
    #     (2, 3)
    # ]

    plt.figure(figsize=(10, 6))
    plt.title('Clustering Results')

    for i in range(len(mat_axis)):
        pmat_cluster_centers_1 = []
        pten_cluster_entries_1 = []
        for cl in range(len(mat_cluster_centers)):
            pmat_cluster_centers_1.append([mat_cluster_centers[cl][mat_axis[i][0]],
                                           mat_cluster_centers[cl][mat_axis[i][1]]])
            pvec_cluster_entries_axis_0 = []
            pvec_cluster_entries_axis_1 = []
            for en in range(len(mat_cluster_entry_indexes[cl])):
                pvec_cluster_entries_axis_0.append(mat_entries[mat_cluster_entry_indexes[cl][en]][mat_axis[i][0]])
                pvec_cluster_entries_axis_1.append(mat_entries[mat_cluster_entry_indexes[cl][en]][mat_axis[i][1]])
            pten_cluster_entries_1.append([pvec_cluster_entries_axis_0, pvec_cluster_entries_axis_1])

        draw_clusters(pmat_cluster_centers_1, pten_cluster_entries_1,
                      vec_param_names[mat_axis[i][0]], vec_param_names[mat_axis[i][1]],
                      plt, i + 1)

    plt.show()
