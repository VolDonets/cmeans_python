import cmeans_math.data_loads as data_loads
import cmeans_math.clustering_by_moving as cl_by_moving
import cmeans_math.clustering_by_evolve as cl_by_evolve
import cmeans_math.clustering_by_move_evolve as cl_by_me
import cmeans_math.new_clustering_by_me as new_cl_me
import accuracies
import cmeans_math.cheat_clustering as cheat_clustering
import cmeans_math.divergences as divergences
import cmeans_math.data_preprocessing as dp

import matplotlib.pyplot as plt
from itertools import combinations

dataset = data_loads.load_from_csv("../test_data/strilets_sales/sales_minmax_scaler_range01.csv",
                                   class_name="cluster", #)
                                   vec_drop_column=['No'])

for i in range(len(dataset.vec_check)):
    dataset.vec_check[i] -= 1

dataset.mat_entries = dp.get_norm_entries(dataset.mat_entries)
dataset.mat_entries = dp.get_updated_data_set(dataset.mat_entries)

print(dataset.mat_entries)


# obj_results0 = new_cl_me.clustering_by_density(mat_entries=dataset.mat_entries,
#                                                var_min_count_clusters=4,
#                                                var_init_count_clusters=20,
#                                                distance="SimpleManhattan")

# obj_results0 = cl_by_me.clustering_by_simple_mahalanobis_density(mat_entries=dataset.mat_entries,
#                                                                  var_min_count_clusters=3,
#                                                                  var_init_count_clusters=20,
#                                                                  # evolve_distance="Mahalanobis")
#                                                                  evolve_distance="Manhattan")

# obj_results0 = cl_by_me.move_evolve_clustering_divergence_loss(mat_entries=dataset.mat_entries,
#                                                                var_init_count_clusters=20,
#                                                                var_min_count_clusters=4,
#                                                                # divergence_func=divergences.kulback_leibler_divergence,
#                                                                divergence_func=divergences.cross_entropy,
#                                                                # distance="Mahalanobis")
#                                                                distance="Manhattan")

obj_results0 = cl_by_me.clustering_by_divergence_density(mat_entries=dataset.mat_entries,
                                                         var_init_count_clusters=20,
                                                         var_count_clusters=3,
                                                         density_func="KulbackLeibler",
                                                         # density_func="CrossEntropy",
                                                         evolve_distance="Manhattan")
                                                         # evolve_distance="Mahalanobis")
                                                         # evolve_distance="KulbackLeibler")
                                                         # evolve_distance="CrossEntropy")


# obj_results = cheat_clustering.show_data_set(mat_entries=dataset.mat_entries,
#                                              vec_check=dataset.vec_check,
#                                              var_init_count_clusters=4)

# obj_results = cheat_clustering.cheat_clear_clustering(mat_entries=dataset.mat_entries,
#                                                       vec_check=dataset.vec_check,
#                                                       var_count_clusters=4)

obj_results = cheat_clustering.cheat_with_noises_clustering(mat_entries=dataset.mat_entries,
                                                            vec_check=dataset.vec_check,
                                                            var_count_clusters=3,
                                                            var_noise=0.07,
                                                            distance="Mahalanobis")
                                                            # distance="Manhattan")

obj_results.vec_cluster_count = obj_results0.vec_cluster_count
obj_results.vec_total_losses = obj_results0.vec_total_losses

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
                            # vec_params=[0, 1, 2])
                            vec_params=[1, 2, 3])
                            # vec_params=[0, 1, 4])

accuracies.draw_3d_clusters(obj_results.mat_cluster_centers, dataset.mat_entries,
                            obj_results.mat_cluster_entry_indexes,
                            vec_param_names=dataset.vec_param_names,
                            # vec_params=[15, 17, 27])
                            # vec_params=[1, 2, 3])
                            vec_params=[2, 3, 4])
                            # vec_params=[7, 10, 11])

accuracies.draw_3d_clusters(obj_results.mat_cluster_centers, dataset.mat_entries,
                            obj_results.mat_cluster_entry_indexes,
                            vec_param_names=dataset.vec_param_names,
                            # vec_params=[13, 19, 30])
                            # vec_params=[0, 2, 3])
                            vec_params=[1, 3, 4])
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

    # mat_axis = [
    #     (0, 1),
    #     (0, 2),
    #     (0, 3),
    #     (1, 2),
    #     (1, 3),
    #     (2, 3)
    # ]

    mat_axis = [
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 3),
        (2, 4),
        (3, 4)
    ]

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
