class ClusteringResults:
    def __init__(self, vec_clusters_count=[], vec_step_number=[], vec_total_losses=[],
                 mat_cluster_centers=[], mat_cluster_entry_indexes=[]):
        self.vec_cluster_count = vec_clusters_count
        self.vec_total_losses = vec_total_losses
        self.mat_cluster_centers = mat_cluster_centers
        self.mat_cluster_entry_indexes = mat_cluster_entry_indexes
        self.vec_step_number = vec_step_number

