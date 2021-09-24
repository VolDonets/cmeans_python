import data_loads as data_loads
import cmeans_math.clustering_by_moving as cl_by_moving
import cmeans_math.clustering_by_evolve as cl_by_evolve
import matplotlib.pyplot as plt

# loading data
iris_data = data_loads.load_iris_data(path="../")

# here is clustering by moving fixed count of cluster centers and mod it
# obj_results = cl_by_moving.clustering_mahalanobis_loss(mat_entries=iris_data.mat_entries, var_count_clusters=3,
#                                                        vec_correct_entry_class=iris_data.vec_check)

####################
# all with density #
####################

# here is clustering by removing cluster centers
obj_results = cl_by_evolve.clustering_by_mahalanobis_density(mat_entries=iris_data.mat_entries,
                                                             var_count_clusters=3,
                                                             vec_correct_entry_class=iris_data.vec_check,
                                                             var_init_count_clusters=10)

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

vec_colors = ['red', 'green', 'blue', 'yellow', 'brown']


def draw_clusters(pmat_cluster_centers, pten_cluster_entries, x_label_name, y_label_name, obj_ax):
    for cl_inx in range(len(pmat_cluster_centers)):
        obj_ax.scatter(pmat_cluster_centers[cl_inx][0], pmat_cluster_centers[cl_inx][1], s=100, c=vec_colors[cl_inx])
        obj_ax.scatter(pten_cluster_entries[cl_inx][0], pten_cluster_entries[cl_inx][1], s=10, c=vec_colors[cl_inx])
    obj_ax.set_title("(" + x_label_name + ", " + y_label_name + ")")
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


mat_entries = iris_data.mat_entries
vec_param_names = iris_data.vec_param_names
mat_cluster_centers = obj_results.mat_cluster_centers
mat_cluster_entry_indexes = obj_results.mat_cluster_entry_indexes

mat_axis = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
fig, ((ax00, ax01, ax02), (ax10, ax11, ax12)) = plt.subplots(2, 3)
vec_obj_axis = [ax00, ax01, ax02, ax10, ax11, ax12]

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
                  vec_obj_axis[i])

fig.tight_layout()
plt.show()
