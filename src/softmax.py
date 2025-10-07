import numpy as np


def softmax_loss(W, X, y, reg=0):
    """
    Softmax loss function as a vectorized implementation.

    Inputs:
    - W: (D, C) weight matrix
    - X: (N, D) data batch
    - y: (N,) labels where y[i] in [0, C-1]
    - reg: regularization strength

    Returns:
    - loss: scalar
    - dW: (D, C) gradient of loss w.r.t. W
    """
    N = X.shape[0]

    scores = X.dot(W)

    # The idea of numerical stability is to shift scores so max is 0 (does not change softmax output)
    scores_stable = scores - np.max(scores, axis=1, keepdims=True)

    exp_scores = np.exp(scores_stable)
    softmax_probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # Cross-entropy loss: -log(probability of correct class)
    correct_class_probs = softmax_probs[np.arange(N), y]
    data_loss = -np.sum(np.log(correct_class_probs)) / N
    reg_loss = 0.5 * reg * np.sum(W * W)
    loss = data_loss + reg_loss

    # Gradient: (predicted_prob - true_label) weighted by input
    y_one_hot = np.zeros_like(softmax_probs)
    y_one_hot[np.arange(N), y] = 1
    dscores = softmax_probs - y_one_hot
    dW = X.T.dot(dscores) / N + reg * W

    return loss, dW