import numpy as np
from medmnist import BloodMNIST


def load_data_nn(size=28, subsample_train=1000, subsample_val=1000, seed=0):
    """
    Load and preprocess data for neural network training.
    Uses mean centering and /255 scaling (different from linear classifiers).
    Does NOT add bias column since NN has separate bias parameters.
    """
    np.random.seed(seed)

    # Load splits
    tr = BloodMNIST(split='train', download=True, size=size)
    va = BloodMNIST(split='val', download=True, size=size)
    te = BloodMNIST(split='test', download=True, size=size)

    X_train = tr.imgs
    y_train = tr.labels.flatten()
    X_val = va.imgs
    y_val = va.labels.flatten()
    X_test = te.imgs
    y_test = te.labels.flatten()

    # Shuffle up training data
    train_data = np.random.permutation(X_train.shape[0])
    X_train = X_train[train_data]
    y_train = y_train[train_data]

    # Subsample data
    if subsample_train is not None:
        X_train = X_train[:subsample_train]
        y_train = y_train[:subsample_train]

    if subsample_val is not None:
        X_val = X_val[:subsample_val]
        y_val = y_val[:subsample_val]

    # Flatten images to vectors
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_val = X_val.reshape(X_val.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    # Ensure images are float32
    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # Preprocessing: subtract the mean image
    # first: compute the image mean based on the training data
    mean_image = np.mean(X_train, axis=0).astype(np.float32)

    # second: subtract the mean image from train and test data
    X_train = X_train - mean_image
    X_val = X_val - mean_image
    X_test = X_test - mean_image

    X_train /= 255.0
    X_val /= 255.0
    X_test /= 255.0

    return X_train, y_train, X_val, y_val, X_test, y_test


class FullyConnectedNN:
    def __init__(self, layers, reg_strength=0.01, loss='softmax', seed=42):
        """
        layers: List where each element represents the number of nodes in that layer.
        reg_strength: L2 regularization strength
        loss: 'softmax' or 'hinge'
        """
        np.random.seed(seed)
        self.layers = layers
        self.reg_strength = reg_strength
        self.loss_type = loss
        self.params = self._initialize_weights(layers)

    def _initialize_weights(self, layers):
        """
        Initialize weights and biases for each layer
        """
        params = {}
        for i in range(1, len(layers)):
            params['W' + str(i)] = np.random.randn(layers[i - 1], layers[i]) * 0.01
            params['b' + str(i)] = np.zeros((1, layers[i]))
        return params

    def relu(self, Z):
        """
        ReLU (Rectified Linear Unit) activation function.
        Used in hidden layers because it:
        - Introduces non-linearity (allows network to learn complex patterns)
        - Avoids vanishing gradient problem (unlike sigmoid/tanh)
        - Computationally efficient
        """
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        """
        Derivative of ReLU for backpropagation.
        Simply 1 if Z > 0, else 0
        """
        return Z > 0

    def softmax(self, Z):
        """
        Softmax activation for output layer in multi-class classification.
        Converts raw scores to probabilities that sum to 1.

        Numerical stability trick: subtract max before exp to prevent overflow
        This doesn't change the output because:
        softmax(Z) = exp(Z) / sum(exp(Z))
                   = exp(Z - max(Z)) / sum(exp(Z - max(Z)))
        """
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def softmax_loss(self, A, y):
        m = y.shape[0]
        p = self.softmax(A)
        log_likelihood = -np.log(p[range(m), y])
        loss = np.sum(log_likelihood) / m
        return loss

    def hinge_loss(self, A, y):
        m = y.shape[0]
        correct_class_scores = A[range(m), y].reshape(-1, 1)
        margins = np.maximum(0, A - correct_class_scores + 1)
        margins[range(m), y] = 0
        loss = np.sum(margins) / m
        return loss

    def compute_loss(self, A, y):
        if self.loss_type == 'softmax':
            return self.softmax_loss(A, y) + self._l2_regularization()
        elif self.loss_type == 'hinge':
            return self.hinge_loss(A, y) + self._l2_regularization()

    def _l2_regularization(self):
        reg_loss = 0
        for i in range(1, len(self.layers)):
            reg_loss += np.sum(np.square(self.params['W' + str(i)]))
        return self.reg_strength * reg_loss / 2

    def forward(self, X):
        cache = {'A0': X}
        A = X
        for i in range(1, len(self.layers)):
            W, b = self.params['W' + str(i)], self.params['b' + str(i)]
            Z = np.dot(A, W) + b
            if i != len(self.layers) - 1:
                A = self.relu(Z)
            else:
                A = Z  # No activation in the output layer (raw scores for loss)
            cache['Z' + str(i)] = Z
            cache['A' + str(i)] = A
        return A, cache

    def backward(self, cache, y):
        """
        Backward pass: compute gradients using chain rule.

        The backpropagation algorithm:
        1. Compute gradient of loss w.r.t. output layer
        2. Propagate this gradient backwards through each layer
        3. At each layer, compute gradients w.r.t. weights and biases
        4. Use chain rule: dL/dW = dL/dA * dA/dZ * dZ/dW

        For softmax + cross-entropy loss, the gradient simplifies to:
        dL/dZ = (predicted_probs - true_labels) / batch_size
        """
        grads = {}
        m = y.shape[0]
        A_last = cache['A' + str(len(self.layers) - 1)]
        if self.loss_type == 'softmax':
            # Softmax + cross-entropy gradient simplifies elegantly
            dA = self.softmax(A_last)
            dA[range(m), y] -= 1  # Subtract 1 from true class probability
            dA /= m
        elif self.loss_type == 'hinge':
            margins = (A_last - A_last[range(m), y].reshape(-1, 1) + 1) > 0
            margins[range(m), y] = 0
            dA = np.where(margins, 1, 0)
            dA /= m

        # Backpropagate through layers using chain rule
        for i in reversed(range(1, len(self.layers))):
            dZ = dA
            A_prev = cache['A' + str(i - 1)]
            # Gradient w.r.t. weights: dL/dW = A_prev^T * dZ + regularization term
            grads['W' + str(i)] = np.dot(A_prev.T, dZ) + self.reg_strength * self.params['W' + str(i)]
            # Gradient w.r.t. biases: dL/db = sum of dZ across batch
            grads['b' + str(i)] = np.sum(dZ, axis=0, keepdims=True)
            if i > 1:
                # Chain rule: multiply by derivative of ReLU activation
                dA = np.dot(dZ, self.params['W' + str(i)].T) * self.relu_derivative(cache['Z' + str(i - 1)])

        return grads

    def update_params(self, grads, learning_rate, v=None, beta1=0.9, beta2=0.999, t=1, optimizer='adam'):
        """
        Updates parameters with chosen optimization method.
        If optimizer is 'momentum' or 'adam', it requires v for velocity and also t for time step in adam.

        SGD (Stochastic Gradient Descent):
        - Simple: W = W - learning_rate * gradient
        - Can oscillate and converge slowly

        Momentum:
        - Momentum:
        - follows these formulas:
        - update rule: w_t+1= w_t - learning_rate * m_t
        - momentum term: m_t = beta * m_t-1 + (1 - beta) * gradient

        Adam (Adaptive Moment Estimation):
        - Maintains both first moment (mean) and second moment (variance) of gradients
        - Adapts learning rate for each parameter
        - Generally converges faster and more reliably
        - Combines benefits of momentum and RMSprop
        """
        for i in range(1, len(self.layers)):
            if optimizer == 'sgd':
                # Standard SGD update
                self.params['W' + str(i)] -= learning_rate * grads['W' + str(i)]
                self.params['b' + str(i)] -= learning_rate * grads['b' + str(i)]
            elif optimizer == 'momentum':
                # Momentum 
                v['m_W' + str(i)] = beta1 * v['m_W' + str(i)] + (1 - beta1) * grads['W' + str(i)]
                v['m_b' + str(i)] = beta1 * v['m_b' + str(i)] + (1 - beta1) * grads['b' + str(i)]

                self.params['W' + str(i)] -= learning_rate * v['m_W' + str(i)]
                self.params['b' + str(i)] -= learning_rate * v['m_b' + str(i)]

            elif optimizer == 'adam':
                # a small constant to prevent division by zero
                eps = 1e-8
                # Update first and second moment estimates for weights
                v['mW' + str(i)] = beta1 * v['mW' + str(i)] + (1 - beta1) * grads['W' + str(i)]
                v['vW' + str(i)] = beta2 * v['vW' + str(i)] + (1 - beta2) * (grads['W' + str(i)] ** 2)

                # Update first and second moment estimates for biases
                v['mb' + str(i)] = beta1 * v['mb' + str(i)] + (1 - beta1) * grads['b' + str(i)]
                v['vb' + str(i)] = beta2 * v['vb' + str(i)] + (1 - beta2) * (grads['b' + str(i)] ** 2)

                # Bias-corrected moment estimates
                mW_hat = v['mW' + str(i)] / (1 - beta1 ** t)
                vW_hat = v['vW' + str(i)] / (1 - beta2 ** t)

                mb_hat = v['mb' + str(i)] / (1 - beta1 ** t)
                vb_hat = v['vb' + str(i)] / (1 - beta2 ** t)

                # Parameter updates
                self.params['W' + str(i)] -= learning_rate * mW_hat / (np.sqrt(vW_hat) + eps)
                self.params['b' + str(i)] -= learning_rate * mb_hat / (np.sqrt(vb_hat) + eps)

    def train(self, X, y, X_val=None, y_val=None, learning_rate=0.01, reg=0.01,
              num_iters=200, batch_size=64, optimizer='adam', print_every=100):
        """
        Trains the model using the chosen optimizer.
        """
        self.reg_strength = reg
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train // batch_size, 1)

        loss_history = []
        train_acc_history = []
        val_acc_history = []

        v = None
        if optimizer == 'momentum':
            v = {}
            for i in range(1, len(self.layers)):
                # moving average of gradients (m_t), NOT velocity
                v['m_W' + str(i)] = np.zeros_like(self.params['W' + str(i)])
                v['m_b' + str(i)] = np.zeros_like(self.params['b' + str(i)])

        elif optimizer == 'adam':
            v = {'t': 0}  # timestep for bias correction
            for i in range(1, len(self.layers)):
                # first moment (m) and second moment (v) for W and b
                v['mW' + str(i)] = np.zeros_like(self.params['W' + str(i)])
                v['vW' + str(i)] = np.zeros_like(self.params['W' + str(i)])
                v['mb' + str(i)] = np.zeros_like(self.params['b' + str(i)])
                v['vb' + str(i)] = np.zeros_like(self.params['b' + str(i)])

        for it in range(num_iters):
            # Sample minibatch
            batch_indices = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]

            A_last, cache = self.forward(X_batch)
            loss = self.compute_loss(A_last, y_batch)
            grads = self.backward(cache, y_batch)

            # pass current time-step to optimizer (used by Adam for bias-correction)
            self.update_params(grads, learning_rate, v=v, optimizer=optimizer, t=it + 1)

            loss_history.append(loss)

            # show results per epoch
            if (it + 1) % iterations_per_epoch == 0:
                train_acc = np.mean(self.predict(X) == y)
                train_acc_history.append(train_acc)
                if X_val is not None and y_val is not None:
                    val_acc = np.mean(self.predict(X_val) == y_val)
                    val_acc_history.append(val_acc)

            if (it + 1) % print_every == 0:
                msg = f"{it + 1}/{num_iters} loss={loss:.4f}"
                if len(train_acc_history):
                    msg += f" train_acc={train_acc_history[-1]:.4f}"
                if len(val_acc_history):
                    msg += f" val_acc={val_acc_history[-1]:.4f}"
                print(msg)

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        A_last, _ = self.forward(X)
        if self.loss_type == 'softmax':
            return np.argmax(A_last, axis=1)
        elif self.loss_type == 'hinge':
            return np.argmax(A_last, axis=1)


# if __name__ == '__main__':
#     # subsample demo
#     X_train, y_train, X_val, y_val, X_test, y_test = load_data_nn(subsample_train=2000, subsample_val=500)
#
#     nn = FullyConnectedNN(layers=[X_train.shape[1], 500, 8], loss='softmax')
#
#     # train the network
#     history = nn.train(X_train, y_train, X_val=X_val, y_val=y_val,
#                        learning_rate=1e-3, reg=1e-3, num_iters=2000,
#                        batch_size=128, optimizer='adam', print_every=200)
#
#     # final accuracies
#     train_acc = np.mean(nn.predict(X_train) == y_train)
#     val_acc = np.mean(nn.predict(X_val) == y_val)
#     print(f'Final Training accuracy={train_acc:.4f}')
#     print(f'Final Validation accuracy={val_acc:.4f}')
#
#     # create plots folder
#     plots_dir = os.path.join(os.path.dirname(__file__), '..', 'plots')
#     os.makedirs(plots_dir, exist_ok=True)
#
#     # Plotting with matplotlib
#     try:
#         import matplotlib.pyplot as plt
#
#         plt.figure(figsize=(8, 4))
#         plt.plot(history['loss_history'])
#         plt.title('Training loss')
#         plt.xlabel('Iteration')
#         plt.ylabel('Loss')
#         plt.grid(True)
#         plt.tight_layout()
#         plt.savefig(os.path.join(plots_dir, '08_nn_training_loss.png'))
#         plt.close()
#
#         if len(history['train_acc_history']) > 0:
#             plt.figure(figsize=(8, 4))
#             plt.plot(history['train_acc_history'], label='train')
#             if len(history['val_acc_history']):
#                 plt.plot(history['val_acc_history'], label='val')
#             plt.title('Accuracy per epoch')
#             plt.xlabel('Epoch')
#             plt.ylabel('Accuracy')
#             plt.legend()
#             plt.grid(True)
#             plt.tight_layout()
#             plt.savefig(os.path.join(plots_dir, '09_nn_training_accuracy.png'))
#             plt.close()
#     except Exception:
#         print('matplotlib not available; skipping plots')