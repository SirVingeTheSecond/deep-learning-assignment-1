import numpy as np


def softmax_loss(W, X, y, reg=0.0):
    """
    Softmax loss function based on a vectorized implementation.

    Inputs:
    - W: An array of shape (D, C) containing weights.
         D = number of features (28*28*3 + 1 = 2353 for BloodMNIST which includes bias)
         C = number of classes (8 for BloodMNIST: basophil, eosinophil, erythroblast, immature granulocytes, lymphocyte, monocyte, neutrophil, platelet)
         Each column W[:, j] represents the weights for class j

    - X: An array of shape (N, D) containing a sliced batch of data.
         N = number of training examples in this batch
         D = number of features per example (2353 for flattened 28x28x3 + bias)
         Each row X[i, :] is one flattened image

    - y: An array of shape (N,) containing training labels
         Each y[i] is an integer from 0 to 7 representing the true class
         y[i] = c means that X[i] has label c, where 0 <= c < C
         This can be a bit hard to understand so here is a small example: y[0] = 3 means first image is class 3 (immature granulocytes)

    - reg: (float) regularization
           Controls how much we penalize large weights to prevent overfitting

    Returns a tuple of:
    - loss: (float) the mean value of the loss function
    - grad: (numpy array) gradient of the loss function with respect to W. It is the same shape as W
    """

    # Init the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)  # dW has same shape as W: (D, C)

    # Number of training examples
    N = X.shape[0]

    # Determine the forward pass
    scores = X.dot(W)  # Shape: (N, C)

    # Here we subtract the max score from each sample to ensure numerical consistency
    scores_stable = scores - np.max(scores, axis=1, keepdims=True)

    # Softmax probabilities
    exp_scores = np.exp(scores_stable)  # Shape: (N, C)
    softmax_probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # Shape: (N, C)

    # Calculate the loss
    # L_i = -log(P(y_i | x_i)) = -log(softmax_probs[i, y_i])
    correct_class_probs = softmax_probs[np.arange(N), y]
    data_loss = -np.sum(np.log(correct_class_probs)) / N

    # Regularization loss
    reg_loss = 0.5 * reg * np.sum(W * W)
    loss = data_loss + reg_loss

    # Calculate the gradient
    # For softmax, dL/dW_j = (1/N) * sum_i (P(j|x_i) - 1[y_i = j]) * x_i
    # This can be vectorized as: dW = (1/N) * X^T * (softmax_probs - y_one_hot)

    # Create one-hot encoding of true labels.
    # One-hot encoding represents the class labels as binary vectors,
    # where only the index corresponding to the true class is set to 1, and all others are 0.
    y_one_hot = np.zeros_like(softmax_probs)
    y_one_hot[np.arange(N), y] = 1

    # Calculate gradient w.r.t. scores
    dscores = softmax_probs - y_one_hot  # Shape: (N, C)

    # Calculate gradient w.r.t. weights
    dW = X.T.dot(dscores) / N  # Shape: (D, C)

    # Now we add the regularization gradient
    dW += reg * W

    return loss, dW
