# Linear Classifier with SVM Loss
## Main
1. Import CI-far 10 data set
2. Split the data into train, Val. And test sets.
3. Reshape the image data into rows
4. Subtract the mean image(all features of each image minus the mean_image)
5. Calculate the SVM Loss of random Weights matrix
```Python
X_train=np.hstack([X.train,np.ones((X_train.shape[0],1))])
And so on
```
$Wx+b =>W'x$
```
We transform Wx+b into W’x by expand the dimension from 3072 to 3073.
Bias value can be covered by W’
```
6. Finish the function: compute _loss_naive
   
   the SVM Loss is defined as follows:

   $L_i=\sum_{j\not =y_i}max(0,s_j-s_{yi}+1)$

   $L=(\sum{L_i})/N$ (N represent N pictures) 

   the Gradient of W:
   
   $dW=dL_i/dw_i$

   $s_j=w[j]x[correct]$
   ```
   for each loss result, increase the correspond label with -X, incorrespond label with X
   ```

```python
def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1] #10 columns of labels
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W) #here we want this to be a one-dimension array(horizontal)A*B=(B.T*A.T).T
    correct_class_score = scores[y[i]] #this pictures scores of ten labels
    for j in xrange(num_classes):
      if j == y[i]: # now we begin to use the formula
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      #Li=max(0,w0*xi-w1*xi+1)+max(0,w2*xi-w1*xi+1)
      #dLi/dwi=-xi(sum)
      #dLi/dwj=xj(sum)
      #both 2 cases when margin>0
      if margin > 0:
        dW[:,y[i]]+=-X[i]
        dW[:,j]+=X[i]
        loss += margin

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W) # it actually did a square (l2 reg)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  dW/=num_train
  dW+=2*reg*W
  return loss, dW
```
7. finish the vectorized:"compute_loss_vectorized", then don't forget to evalute by calculating the Frobenius Norm of the methods' results.
```py
def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_train=X.shape[0]
  scores=X.dot(W) # 使用一次点乘实现
  right_scores=scores[range(num_train),y].reshape(-1,1)
  #隐含了一次循环：for i in range(num_train): add y[i] then reshape
  margin=np.maximum(scores-right_scores+1,0)
  margin[range(num_train),y]=0
  #隐含了一次循环，对正确项的1置0
  loss=np.sum(margin)/num_train+reg*np.sum(W*W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  #所有非0项才存在梯度gradient
  margin[margin>0]=1 # make them BOOL
  margin[range(num_train),y]=-np.sum(margin,axis=1) #统计共有多少个非本label非0项
  dW=np.dot(X.T,margin) #并行完成500个样本对dW的贡献
  dW/=num_train
  dW+=2*reg*W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW

```
8. Stochastic Gredient Descent
   
```
pick part of the set to represent the feature of the whole set.
In this way, we can obtain the gradient much faster.
So that we can multiply it with learning rate to get a better W matrix.
```
```py
class LinearClassifier(object):

  def __init__(self):
    self.W = None

  def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this linear classifier using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) containing training data; there are N
      training samples each of dimension D.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c
      means that X[i] has label 0 <= c < C for C classes.
    - learning_rate: (float) learning rate for optimization.
    - reg: (float) regularization strength.
    - num_iters: (integer) number of steps to take when optimizing
    - batch_size: (integer) number of training examples to use at each step.
    - verbose: (boolean) If true, print progress during optimization.

    Outputs:
    A list containing the value of the loss function at each training iteration.
    """
    num_train, dim = X.shape
    num_classes = np.max(y) + 1 # assume y takes values 0...K-1 where K is number of classes
    if self.W is None:
      # lazily initialize W
      self.W = 0.001 * np.random.randn(dim, num_classes)

    # Run stochastic gradient descent to optimize W
    loss_history = []
    for it in xrange(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # TODO:                                                                 #
      # Sample batch_size elements from the training data and their           #
      # corresponding labels to use in this round of gradient descent.        #
      # Store the data in X_batch and their corresponding labels in           #
      # y_batch; after sampling X_batch should have shape (dim, batch_size)   #
      # and y_batch should have shape (batch_size,)                           #
      #                                                                       #
      # Hint: Use np.random.choice to generate indices. Sampling with         #
      # replacement is faster than sampling without replacement.              #
      #########################################################################
      indexes=np.random.choice(num_train,batch_size)
      X_batch=X[indexes,:]
      y_batch=y[indexes]
      #########################################################################
      #                       END OF YOUR CODE                                #
      #########################################################################

      # evaluate loss and gradient
      loss, grad = self.loss(X_batch, y_batch, reg)
      loss_history.append(loss)

      # perform parameter update
      #########################################################################
      # TODO:                                                                 #
      # Update the weights using the gradient and the learning rate.          #
      #########################################################################
      self.W-=learning_rate*grad
      #########################################################################
      #                       END OF YOUR CODE                                #
      #########################################################################

      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

    return loss_history

  def predict(self, X):
    """
    Use the trained weights of this linear classifier to predict labels for
    data points.

    Inputs:
    - X: A numpy array of shape (N, D) containing training data; there are N
      training samples each of dimension D.

    Returns:
    - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
      array of length N, and each element is an integer giving the predicted
      class.
    """
    y_pred = np.zeros(X.shape[0])
    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Store the predicted labels in y_pred.            #
    ###########################################################################
    result=X.dot(self.W)
    for i in range(result.shape[0]):
      y_pred[i]=np.argmax(result[i])
    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################
    return y_pred
```
9. Utilize the 2 nested for loop to choose a proper Learning Rate and Regularizaion Strength
10. Draw the Plot and visualize the finished W matrix(images of each label)\

9 and 10 can be found in correspond jupyter notebook

## New Stuff / Features
1. Notice the way to calculate Gradient
2. the application of 
```py
margin[range(num_train),y]=0
#############################
margin[y[i]] for i in range(num_train)
#############################
# it pick the labels
```
3. np.random.choice => indexes
   
   X[indexes]=> many random chosen Xs
