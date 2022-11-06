import numpy as np
from fcmeans import FCM
from jax import random

from cmeans_math.covariances import cluster_covariances
from cmeans_math.distances import mahalanobis_distance


class FCM_coef(FCM):
    def __init__(self, mat_X, *args, **kwargs):
        super(FCM_coef, self).__init__(*args, **kwargs)
        self.X = mat_X
        self.n_samples = 0
        self.tensor_covariances = None
        self.mat_cei = None

    def fit(self):
        self.n_samples = self.X.shape[0]
        self.u = random.uniform(key=self.key, shape=(self.n_samples,self.n_clusters))
        self.u = self.u / np.tile(self.u.sum(axis=1)[np.newaxis].T, self.n_clusters)
        for iteration in range(self.max_iter):
            u_old = self.u.copy()
            self.centers = FCM._next_centers(self.X, self.u, self.m)
            self.u = self.__predict(self.X)
            # Stopping rule
            if np.linalg.norm(self.u - u_old) < self.error:
                break

        prediction = self.__predict(self.X).argmax(axis=-1)
        self.mat_cei = []
        for inx in range(self.n_clusters):
            self.mat_cei.append([])

        for inx in range(len(prediction)):
            self.mat_cei[prediction[inx]].append(inx)

        self.tensor_covariances = cluster_covariances(mat_entries=self.X,
                                                      mat_cluster_centers=self.centers,
                                                      mat_cluster_entry_indexes=self.mat_cei)

    def __predict(self, X):
        """
        Parameters
        ----------
        X : array, shape = [n_samples, n_features]
            New data to predict.

        Returns
        -------
        u: array, shape = [n_samples, n_clusters]
            Fuzzy partition array, returned as an array with n_samples rows
            and n_clusters columns.
        """
        temp = FCM._dist(X, self.centers) ** float(2 / (self.m - 1))
        denominator_ = temp.reshape((X.shape[0], 1, -1)).repeat(temp.shape[-1], axis=1)
        denominator_ = temp[:, :, np.newaxis] / denominator_
        return 1 / denominator_.sum(2)

    @property
    def fukuyama_sugeno_index(self):
        if hasattr(self, 'u'):
            vec_x_center = np.sum(self.X, axis=0) / self.X.shape[0]
            mat_centers_norm = self.centers - vec_x_center
            vec_centers_norm_sum = np.sum((mat_centers_norm ** 2), axis=1)
            mat_um = self.u ** self.m
            var_Km = np.sum(mat_um.dot(vec_centers_norm_sum))


            var_Jm = 0.0
            for cl in range(self.n_clusters):
                for inx in range(self.X.shape[0]):
                    var_Jm += mat_um[cl][inx] * np.sum((self.X[inx] - self.centers[cl]) ** 2)

            return var_Jm - var_Km
        else:
            raise ReferenceError("You need to train the model first. You can use `.fit()` method to this.")

    @property
    def partition_coef_exp_sep(self):
        if self.n_clusters == 1:
            return 0.0
        if hasattr(self, 'u'):
            mat_um = self.u ** self.m
            var_min_um = mat_um.min()

            vec_x_center = np.sum(self.X, axis=0) / self.X.shape[0]
            var_betta_t = np.sum((self.centers - vec_x_center) ** 2) / self.n_clusters

            res_0 = self.partition_coefficient / var_min_um
            res_1 = 0.0
            for c in range(self.n_clusters):
                vec_sums = []
                for k in range(self.n_clusters):
                    if k != c:
                        vec_sums.append(np.sum(self.centers[c] - self.centers[k]) ** 2)
                vec_sums = np.array(vec_sums)
                res_1 += np.exp(-vec_sums.min() / var_betta_t)

            return res_0 - res_1
        else:
            raise ReferenceError("You need to train the model first. You can use `.fit()` method to this.")

    @property
    def mahalanobis_inverse_coefficient(self):
        if hasattr(self, 'u'):
            val_coef = 0.0
            for cl in range(self.n_clusters):
                val_cl_coef = 0.0
                for el in self.mat_cei[cl]:
                    val_cl_coef += mahalanobis_distance(self.X[el], self.centers[cl],
                                                        self.tensor_covariances[cl]) / self.u[el][cl]
                if len(self.mat_cei[cl]) != 0:
                    val_coef += val_cl_coef / len(self.mat_cei[cl])
            return val_coef / self.n_clusters
        else:
            raise ReferenceError("You need to train the model first. You can use `.fit()` method to this.")

    @property
    def mahalanobis_coefficient(self):
        val_coef = 0.0
        for cl in range(self.n_clusters):
            val_cl_coef = 0.0
            for el in self.mat_cei[cl]:
                val_cl_coef += mahalanobis_distance(self.X[el], self.centers[cl],
                                                    self.tensor_covariances[cl])
            if len(self.mat_cei[cl]) != 0:
                val_coef += val_cl_coef / len(self.mat_cei[cl])
        return val_coef / self.n_clusters

    @property
    def kulback_leibler_coefficient(self):
        if hasattr(self, 'u'):
            var_klc = 0.0
            for el in range(self.X.shape[0]):
                var_klc_el = 0.0
                for cl in range(self.n_clusters):
                    for cl_r in range(self.n_clusters):
                        var_klc_el += self.u[el][cl] * np.log(self.u[el][cl] / (self.u[el][cl_r] + 0.000000001))
                var_klc -= var_klc_el / self.n_clusters

            return var_klc / self.n_clusters
        else:
            raise ReferenceError("You need to train the model first. You can use `.fit()` method to this.")


if __name__ == '__main__':
    import cmeans_math.data_loads as data_loads
    import cmeans_math.data_preprocessing as dp
    import matplotlib.pyplot as plt

    dataset = data_loads.load_iris_data("../")
    dataset.mat_entries = dp.get_norm_entries(dataset.mat_entries)
    dataset.mat_entries = dp.get_updated_data_set(dataset.mat_entries)

    X = np.array(dataset.mat_entries)
    Y = np.array(dataset.vec_check)

    fsi = []
    pc = []
    pec = []
    pcaes = []
    mah = []
    mah_inv = []
    klc = []
    for i in range(1, 21):
        cmeans_tst = FCM_coef(mat_X=X, n_clusters=i, max_iter=300, random_state=1)
        cmeans_tst.fit()
        fsi.append(cmeans_tst.fukuyama_sugeno_index)
        pc.append(cmeans_tst.partition_coefficient)
        pec.append(cmeans_tst.partition_entropy_coefficient)
        pcaes.append(cmeans_tst.partition_coef_exp_sep)
        mah.append(cmeans_tst.mahalanobis_coefficient)
        mah_inv.append(cmeans_tst.mahalanobis_inverse_coefficient)
        klc.append(cmeans_tst.kulback_leibler_coefficient)
        print('OK', i)

    plt.plot(range(1, 21), klc)
    plt.xlabel('Clusters count')
    plt.ylabel('Total loss')
    plt.title('My cluster compactness')
    plt.show()

    plt.plot(range(1, 21), mah)
    plt.xlabel('Clusters count')
    plt.ylabel('Total loss')
    plt.title('My cluster compactness')
    plt.show()

    plt.plot(range(1, 21), mah_inv)
    plt.xlabel('Clusters count')
    plt.ylabel('Total loss')
    plt.title('My cluster compactness')
    plt.show()

    plt.plot(range(1, 21), pcaes)
    plt.xlabel('Clusters count')
    plt.ylabel('Total loss')
    plt.title('My cluster compactness')
    plt.show()

    plt.plot(range(1, 21), fsi)
    plt.xlabel('Clusters count')
    plt.ylabel('Total loss')
    plt.title('My cluster compactness')
    plt.show()

    plt.plot(range(1, 21), pc)
    plt.xlabel('Clusters count')
    plt.ylabel('Total loss')
    plt.title('My cluster compactness')
    plt.show()

    plt.plot(range(1, 21), pec)
    plt.xlabel('Clusters count')
    plt.ylabel('Total loss')
    plt.title('My cluster compactness')
    plt.show()
