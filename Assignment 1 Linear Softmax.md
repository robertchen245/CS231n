# Linear Classifier with Softmax Loss
## Main
1. Import CI-far 10 data set
2. Split the data into train, validation, test, dev
3. Reshape them into rows
4. Preprocess: subtract the mean image(do the same minus in SVM)
5. Consider the bias along with Weight matrix.
   
$Wx+b=>W'x$
```
3072->3073
```
6. Finish the function: softmax_loss_naive() with nested loops
```
Here are the definitions and formulas of Softmax
multinomial logistic regression:
```
$P(Y=k|X=X_i)={\Large e^{s_y}\over \sum{e^{s_j}}}$

$Li=-\log(P)=-\log{\LARGE e^{s_y}\over \sum{e^{s_j}}}, (\log = \ln)$
```
When all scores are low (not relevant)
```
$s\rightarrow0$

$e^s\rightarrow1$

$Li=-\log(P)\rightarrow-\log{1\over c}\rightarrow\log c$

$L={\sum Li \over C}+\alpha Reg$
```
And to ensure the precision and avoid the overflow, we can perform a mathematical trick:
```
$\LARGE {e^{s_y}\over \sum{e^{s_j}}} ={Ke^{s_y}\over \sum{Ke^{s_j}}}={e^{s_y+logK}\over \sum{e^{s_j+logK}}}$
```
We make K the max of score, so that all the exponential term can be minimized.
```
$\LARGE K=-e^{max\{scores\}}$
```
Notice that there must be one term that is not zero:
```
$\LARGE e^0 = 1$
```
The Loss is relatively simple to calculate, let's deduce the Gradient expression:
```
$\Large Grad_i={\partial L_i \over \partial W}={\partial L_i \over \partial S}{\partial S \over \partial W}$

$\Large {\partial L_i \over \partial S}={\partial \log \sum e^{S_j} -\partial \log e^{s_y} \over \partial S_j}={\partial \log (others+e^S) \over \partial S}-{\partial S_{y_i} \over \partial Sj}={e^{s_j}\over \sum{e^{s_j}}}+{\partial S_{y_i} \over \partial Sj}$

$\Large When\ j\not ={y_i},\ Grad_i={e^{s_j}\over \sum{e^{s_j}}}$ 

$\Large When\ j={y_i},\ Grad_i={e^{s_j}\over \sum{e^{s_j}}}-1$ 
```py
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
  nums_train=X.shape[0]
  nums_classes=W.shape[1]
  for i in xrange(nums_train):
    score=X[i].dot(W)
    score-=np.max(score)
    correct_class_score=np.exp(score[y[i]])
    sum=np.sum(np.exp(score)) # the sum of e^
    for j in xrange(nums_classes):
      if j!=y[i]:
        dW[:,j]+=X[i]*(np.exp(score[j])/sum) #when j!=y
      else:
        dW[:,j]+=X[i]*((np.exp(score[j])/sum)-1) #when j=y
    loss+=-np.log(correct_class_score/sum)
  loss/=nums_train
  loss+=reg*np.sum(W*W)
  dW/=nums_train
  dW+=2*reg*W # don't forget to add up the regression
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
```
7. Finish the function: softmax_loss_vectorized()

$Wx=Score$
```
Extract highest scores in each row. Then make the scores to be exponential(don't forget to minus the exponential terms of highest score)
```
$\Large When\ j\not ={y_i},\ Grad_i={e^{s_j}\over \sum{e^{s_j}}}$ 

$\Large When\ j={y_i},\ Grad_i={e^{s_j}\over \sum{e^{s_j}}}-1$ 
```
Preprocess the scores matrix, to make it ready to calculate Grad.
```
```py

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
  nums_train=X.shape[0]
  nums_classes=W.shape[1]
  scores=X.dot(W)
  scores-=np.max(scores,axis=1).reshape(-1,1) #平移化的score矩阵，按每一行最大元素进行平移
  correct_scores=np.exp(scores[range(nums_train),y].reshape(1,-1))
  scores=np.exp(scores)
  loss=np.sum(-np.log(correct_scores/np.sum(scores,axis=1)))
  scores/=np.sum(scores,axis=1).reshape(-1,1)
  scores[range(nums_train),y]-=1
  dW=X.T.dot(scores)
  loss/=nums_train
  loss+=reg*np.sum(W*W)
  dW/=nums_train
  dW+=2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
```
8. SGD to get the best Weight matrix
9. Cross-validation to get the best learning rate and regularizaion
10. Visualize the Weight
```
8, 9, 10 has been shown in Jupyter notebook
```
## New Stuff / Features
1. The application of np.mean(train==predict) to calculate the accuracy !!!
```py
Accuracy=np.mean(y_predict==y_train)
```
2. The way to deduce Gradient expression.
3. Review the gradient of SVM, when j!=y , W(y) still has Xi every loop .$\sum Xi$, which is slightly different from Softmax Gradient
