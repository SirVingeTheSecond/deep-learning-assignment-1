import numpy as np

def svm_loss(W,X,y,reg=0):
  #Image count
  N = X.shape[0]

  scores = X.dot(W) # (N, C)
  print(scores.shape)
  # reshape (-1, 1) turns it from an array of 
  # length 5000 to a matrix of size (N, 1) 
  # (-1 means to keep same length as original shape)
  s_y = scores[np.arange(N), y].reshape(-1, 1) # (N, 1)
  print(s_y.shape)

  #Calculate margins for each score
  margins = np.maximum(0, scores - s_y + 1) # (N, C)
  #Do not count the correct label, just setting it to 0
  margins[np.arange(N),y] = 0

  loss = np.sum(margins) / N #Average loss
  print(loss)

  #TODO: Regularization loss
  # l2 regularization
  if reg == 2 :
    # square the weight of each entry * sum of all rows * sum of columns
    l2_regularization_loss = np.sum(W**2)
    accumulated_loss = loss+l2_regularization_loss
    print(f" accumulated_loss{accumulated_loss}")
    return accumulated_loss

  return loss