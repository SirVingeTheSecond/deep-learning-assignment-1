import numpy as np

################################################
###### implementering af kNN classifier ########
################################################
class KNearestNeighbor:
  def __init__(self) -> None:
    self.Y_train = None
    self.X_train = None

  def train(self, X_train, Y_train):
    self.X_train = X_train
    self.Y_train = Y_train

  def calculate_distances(self, X_test):
    #Initialize distances array
    distances = np.zeros((X_test.shape[0], self.X_train.shape[0]))

    #Get distances to each image in training per test image
    for i in range(X_test.shape[0]):
      distances[i] = np.sqrt(np.sum(np.square(self.X_train - X_test[i]), axis = 1))

    return distances

  def predict(self, X_test, k):
    distances = self.calculate_distances(X_test)
    predLabel = np.zeros(X_test.shape[0])
    for i in range(X_test.shape[0]):
      #Get indicies for k nearest neighbors
      args = np.argpartition(distances[i], k)[:k]
      #Get labels of k nearest neighbors
      labels = self.Y_train[args]
      #Find the most common label
      values, counts = np.unique(labels, return_counts=True)
      predLabel[i] = labels[np.argmax(counts)]

    return predLabel