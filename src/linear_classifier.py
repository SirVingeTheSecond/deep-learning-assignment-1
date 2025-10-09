import numpy as np
from softmax import softmax_loss
from svm import svm_loss
from data import load_data

class LinearClassifier:
    """
    Linear classifier that can use either SVM or Softmax loss.
    """

    def __init__(self, input_dim, num_classes, loss_type='softmax'):
        # Init weights to small random values
        # Shape: (D, C) where D=input_dim, C=num_classes
        self.W = np.random.randn(input_dim, num_classes) * 0.0001
        self.loss_type = loss_type

    def train(self, X, y, X_val=None, y_val=None, learning_rate=1e-3, reg=1e-5, num_iters=200, batch_size=200,
              print_every=100):
        num_train = X.shape[0]
        num_batches = int(np.ceil(num_train / batch_size))

        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            # Sample minibatch
            x_batches = np.array_split(X, num_batches)
            y_batches = np.array_split(y, num_batches)
            avg_loss = 0
            for x_batch, y_batch in zip(x_batches, y_batches):
                avg_loss = 0
                if self.loss_type == 'softmax':
                    loss, gradient = softmax_loss(self.W, x_batch, y_batch, reg)
                elif self.loss_type == 'svm':
                    loss, gradient = svm_loss(self.W, x_batch, y_batch, reg)

                #Apply combined loss
                self.W -= gradient * learning_rate
                avg_loss += loss
            
            loss_history.append(avg_loss / num_batches)

            if (it + 1) % print_every == 0:
                print(str(it + 1) + "/" + str(num_iters) + " " + str(avg_loss / num_batches))

            # Check accuracy every epoch
            train_pred = self.predict(x_batch)
            train_acc = np.mean(train_pred == y_batch)
            train_acc_history.append(train_acc)

            if X_val is not None and y_val is not None:
                val_pred = self.predict(X_val)
                val_acc = np.mean(val_pred == y_val)
                val_acc_history.append(val_acc)

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    # def train(self, X, y, X_val=None, y_val=None, learning_rate=1e-3, reg=1e-5, num_iters=200, batch_size=200, print_every=100):
    #     for it in range(num_iters):
    #         # We are not doing any mini-batching here
    #         # ToDo: Implement mini-batching
    #         x_batch = X
    #         y_batch = y
    #
    #         loss, gradient = None, None
    #
    #         if self.loss_type == 'softmax':
    #             loss, gradient = softmax_loss(self.W, x_batch, y_batch, reg)
    #         elif self.loss_type == 'svm':
    #             loss, gradient = svm_loss(self.W, x_batch, y_batch, reg)
    #
    #         self.W -= gradient * learning_rate
    #
    #         print(str(it + 1) + "/" + str(num_iters) + " " + str(loss))

    def predict(self, X):
        scores = X.dot(self.W)

        y_pred = np.argmax(scores, axis=1)

        return y_pred

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

# X_train, y_train, x_val, y_val, X_test, y_test = load_data(size=28, subsample_train=5000)
# classifer = LinearClassifier(28*28*3+1, 8, 'svm')
#
# classifer.train(X_train, y_train, x_val, y_val, num_iters=500)
#
# pred = classifer.predict(X_test)
#
# print(pred)
#
# accuracy = np.mean(pred == y_test) * 100
# print(accuracy)