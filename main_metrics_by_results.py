from cmeans_math import accuracies as ac

mat_cluster_entry_indexes = [[0, 2, 3, 4, 5, 6, 7, 8, 17, 20],
                             [9, 10, 11, 12, 13, 15],
                             [14],
                             [1, 16, 18, 19, 21]]
vec_correct_entry_class = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3]

acc = ac.accuracy(mat_cluster_entry_indexes=mat_cluster_entry_indexes,
                  vec_correct_entry_class=vec_correct_entry_class)
print('accuracy =', acc)

mat_confusion = ac.confusion_matrix(mat_cluster_entry_indexes=mat_cluster_entry_indexes,
                                    vec_correct_entry_class=vec_correct_entry_class)

print("Confusion")
for vec_confusion in mat_confusion:
    print(vec_confusion)

ac.print_cluster_num(mat_cluster_entry_indexes=mat_cluster_entry_indexes,
                     vec_correct_entry_class=vec_correct_entry_class)

ac.draw_roc_curve(mat_cluster_entry_indexes=mat_cluster_entry_indexes,
                  vec_correct_entry_class=vec_correct_entry_class,
                  var_count_cluster=4)

ac.draw_multilabel_roc_curve(mat_cluster_entry_indexes=mat_cluster_entry_indexes,
                             vec_correct_entry_class=vec_correct_entry_class,
                             var_count_cluster=4)
