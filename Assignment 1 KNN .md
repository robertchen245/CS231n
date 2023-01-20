# K Nearest Neighbor 
## Main
1. Import CI-far 10 data set
2. Create an KNN Classifier
3. Implement compute L2 distance (2 loops, 1 loops, no loop)
Based on following equation (2 loop first)

$L2=\sqrt{\sum \left( a_{test}-a_{train}\right) ^{2}}$
```Python
def compute_distances_two_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a nested loop over both the training data and the 
    test data.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the Euclidean distance between the ith test point and the jth training
      point.
    """
    num_test = X.shape[0] #test num is 500
    num_train = self.X_train.shape[0]# train num is 5000
    dists = np.zeros((num_test, num_train)) #create a (500,5000) 
    for i in xrange(num_test):
      for j in xrange(num_train):
        #####################################################################
        # TODO:                                                             #
        # Compute the l2 distance between the ith test point and the jth    #
        # training point, and store the result in dists[i, j]. You should   #
        # not use a loop over dimension.                                    #
        #####################################################################
        dists[i,j]=np.sqrt(np.sum((X[i]-self.X_train[j])**2))
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
    return dists
```
4. Calling this method to get the dists matrix, label each image when K=1.
   the accurracy is around 27%
5. set K=5, accurracy slightly improved.
6. implement 1 loop compute L2 distance according to the  broadcast rules.
```python
def compute_distances_one_loop(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a single loop over the test data.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in xrange(num_test):
      #######################################################################
      # TODO:                                                               #
      # Compute the l2 distance between the ith test point and all training #
      # points, and store the result in dists[i, :].                        #
      #######################################################################
      dists[i,:]=np.sqrt(np.sum((X[i]-self.X_train)**2,axis=1)) #X被广播原则推广到5000行 列元素不变
      #######################################################################
      #                         END OF YOUR CODE                            #
      #######################################################################
    return dists
```
7. Use the Frobenius Norm to ensure our vectorization implementation is correct.
(namely, calculate the L2 distance of $(dists-dists_{one})$)
8. Fully vectorization of implementation: no loop
ideas:
```
thinking about the L2 distance: 
```  
$L2=\sqrt{\sum \left( a_{test}-a_{train}\right) ^{2}}$
```
notice that it is complete square formula:
```
$(a-b)^2=a^2+b^2-2ab$

=>

$L2=\sqrt{\sum (a_{test}^2+a_{train}^2-2a_{test}a_{train})}$
```
so the preprocess is: Square all tests' features then sum by column. Same operation for trains'.
then utilize the broadcast rules.
for the third part, Tranposition then use broadcast to get the dot product(2ab)
the dist is ready for label
```
```python
def compute_distances_no_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train)) # try (a-b)^2 = a^2+b^2-2ab
    #########################################################################
    # TODO:                                                                 #
    # Compute the l2 distance between all test points and all training      #
    # points without using any explicit loops, and store the result in      #
    # dists.                                                                #
    #                                                                       #
    # You should implement this function using only basic array operations; #
    # in particular you should not use functions from scipy.                #
    #                                                                       #
    # HINT: Try to formulate the l2 distance using matrix multiplication    #
    #       and two broadcast sums.                                         #
    #########################################################################
    sum1=np.sum(X**2,axis=1)
    sum1=sum1.reshape(-1,1)
    sum2=np.sum(self.X_train**2,axis=1)
    sum2=sum2.reshape(1,-1)
    dot1=np.dot(X,self.X_train.T)
    sum3=sum1+sum2
    dists=np.sqrt(sum3-2*dot1)
    #########################################################################
    #                         END OF YOUR CODE                              #
    #########################################################################
    return dists
```
9. Crose-validation: codes are already in the jupyter notebook.
## New Stuff / Features
1. Broadcast rules in NumPy
2. Vectorizaition to reduce time complexity (utilization of complete square formula)
3. vstack / hstack 
4. notice the shape (x,y) means x rows, y columns, but (x,) means 1 rows x columns(only one dimension)
