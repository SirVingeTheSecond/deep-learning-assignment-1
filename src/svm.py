import numpy as np

def svm_loss(W,X,y,reg=0.0):
  """
  SVM Loss function, vectorized implementation

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

    - reg: Regulization strenght (float)
  
  Returns tuple containing:
    - loss: (float) the mean value of the loss function
    - grad: (numpy array) gradient of the loss function with respect to W. It is the same shape as W
  """
  #Image count
  N = X.shape[0]

  scores = X.dot(W) # (N, C)
  # reshape (-1, 1) turns it from an array of 
  # length 5000 to a matrix of size (N, 1) 
  # (-1 means to keep same length as original shape)
  s_y = scores[np.arange(N), y].reshape(-1, 1) # (N, 1)

  #Calculate margins for each score
  margins = np.maximum(0, scores - s_y + 1) # (N, C)
  #Do not count the correct label, just setting it to 0
  margins[np.arange(N),y] = 0

  loss = np.sum(margins) / N #Average loss
  #loss += 0.5 * reg * np.sum(W**2)
  
  #Gradient
  #dL_i/dW_j = 𝟙(s_i - s_yi > 0)*x_i
  binary = (margins > 0).astype(float) 
  exceeded = np.sum(binary, axis=1) #Number of classes that exceeded the margin

  binary[np.arange(N), y] = -exceeded #Will contain 1 on every incorrect class

  dW = X.T.dot(binary) / N  # Shape: (D, C)
  #dW += reg * W #Regulization gradient

  return loss, dW