from scipy.spatial.distance import cdist
import numpy as np

################################################
###### implementering af kNN classifier ########
################################################
class KNearestNeighbor:
    def __init__(self, metric='L2'):
        self.Y_train = None
        self.X_train = None
        self.metric = metric

    def train(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train

    def calculate_distances(self, X_test, metric=None):
        if metric is None:
            metric = self.metric

        # Calculate distances between test and training points
        n_test = X_test.shape[0]
        n_train = self.X_train.shape[0]
        distances = np.zeros((n_test, n_train))

        if metric == 'L2':
            # Vectorized L2 (Euclidian) distance (fra slides)
            # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a*b
            test_sq = np.sum(X_test ** 2, axis=1, keepdims=True)
            train_sq = np.sum(self.X_train ** 2, axis=1)
            cross_term = np.dot(X_test, self.X_train.T)

            # RuntimeWarning: invalid value encountered in sqrt distances = np.sqrt(test_sq + train_sq - 2 * cross_term)
            squared_distances = test_sq + train_sq - 2 * cross_term
            squared_distances = np.maximum(squared_distances, 0) # Ensure non-negative value
            distances = np.sqrt(squared_distances)

        elif metric == 'L1':
            # L1 (Manhattan) distance
            distances = cdist(X_test, self.X_train, metric='cityblock')
            # Slow as hell due to not being vectorized
            # for i in range(n_test):
            #     distances[i] = np.sum(np.abs(self.X_train - X_test[i]), axis=1)
        else:
            raise ValueError("Distance metric must be 'L1' or 'L2'")

        return distances

    def predict(self, X_test, k, metric=None):
        distances = self.calculate_distances(X_test, metric)
        predLabel = np.zeros(X_test.shape[0])

        for i in range(X_test.shape[0]):
            # Get indices for k nearest neighbors
            k_nearest_indices = np.argpartition(distances[i], k)[:k]
            # Get labels of k nearest neighbors
            k_nearest_labels = self.Y_train[k_nearest_indices]
            # most common label (majority vote)
            values, counts = np.unique(k_nearest_labels, return_counts=True)
            #predLabel[i] = labels[np.argmax(counts)]
            predLabel[i] = values[np.argmax(counts)]

        return predLabel.astype(int)