import numpy as np
from softmax import softmax_loss
from svm import svm_loss

class LinearClassifier:
    """
    Linear classifier that can use either SVM or Softmax loss.
    """

    def __init__(self, input_dim, num_classes, loss_type='softmax'):
        # Init weights to small random values
        # Shape: (D, C) where D=input_dim, C=num_classes
        self.W = np.random.randn(input_dim, num_classes) * 0.0001
        self.loss_type = loss_type

    def train(self, X, y, X_val=None, y_val=None, learning_rate=1e-3, reg=1e-5, num_iters=1000, batch_size=200):
        # ToDo
        pass

    def predict(self, X):
        # ToDo
        pass

    def loss(self, X, y, reg=0.0):
        """
        Calculate the loss on the given dataset.

        Inputs:
        - X: A numpy array of shape (N, D) containing data
        - y: A numpy array of shape (N,) containing labels
        - reg: (float) regularization

        Returns:
        - loss: (float) the loss value
        """
        if self.loss_type == 'softmax':
            loss, _ = softmax_loss(self.W, X, y, reg)
        elif self.loss_type == 'svm':
            loss, _ = svm_loss(self.W, X, y, reg)
        else:
            raise ValueError(f'Unknown loss_type: {self.loss_type}')

        return loss