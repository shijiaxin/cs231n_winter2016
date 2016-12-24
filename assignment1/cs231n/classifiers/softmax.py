import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  C=np.max(y)+1;
  N=X.shape[0];
  exp_matrix=np.exp(X.dot(W));
  row_sum=exp_matrix.sum(axis=1);
  row_correct=exp_matrix[np.arange(N),y];
  loss=-1.0/N*np.sum(np.log(row_correct/row_sum));
  loss += 0.5 * reg * np.sum(W * W);

  # Middle = X.dot(W);
  dMiddle=np.zeros((N,C));
  dMiddle[np.arange(N),y]-=1;
  dMiddle+=exp_matrix/(row_sum.reshape(N,1));
  dMiddle/=N;
  dW=X.T.dot(dMiddle);
  dW += reg * W;
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  C=np.max(y)+1;
  N=X.shape[0];
  scores=X.dot(W);
  exp_scores=np.exp(scores-scores.max(axis=1,keepdims=True));
  probs=exp_scores/exp_scores.sum(axis=1,keepdims=True);
  loss=-1.0/N*np.sum(np.log(probs[np.arange(N),y]));
  loss += 0.5 * reg * np.sum(W * W);

  dscores=probs;
  dscores[np.arange(N),y]-=1;
  dscores/=N;
  dW=X.T.dot(dscores);
  dW += reg * W;
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
