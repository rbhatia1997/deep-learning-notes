# Introduction
This will contain concepts and coding tools used for the first part of the tutorial. 

## Python Basics with Numpy 
Numpy is used by a large group of people and there are important functions for use with Python. Because in deep learning we mainly use matrices and vectors, we use the numpy library more. Now, implementing the function sigmoid_grad() to compute the gradient of the sigmoid function with respect to its input x. The formula is: $$sigmoid\_derivative(x) = \sigma'(x) = \sigma(x) (1 - \sigma(x))\tag{2}$$
You often code this function in two steps: 
1. Set s to be the sigmoid of x. You might find your sigmoid(x) function useful.
2. Compute $\sigma'(x) = s(1-s)$

This is useful for computing the gradient to optimize loss functions using backpropagation. In deep learning, often np.shape and np.reshape() are used to get the shape of a matrix/vector and to reshape a matrix into some other dimension. Often, when reading a 3D array of shape, you want to unroll or reshape the image to a 1D vector by doing the following: ((length X height X 3),1). You can reshape an array v of shape (a,b,c) into a vector of shape (a*b, c) by doing the following: ```v = v.reshape((v.shape[0]*v.shape[1], v.shape[2]))```. Reshaping a 3D image to a 1D image looks like the following: ```v = image.reshape((image.shape[0]*image.shape[1]*image.shape[2],1))```. 

Another technique used in ML and Deep Learning is normalizing the data. Leads to better performance because gradient descent converges faster after normalization. Normalization is defined as changing x to x/(abs(x)) or dividing each row vector of x by its norm. The norm is defined as the sqrt(element1^2 + element2^2 + ...). Calculating the norm is done via: ```np.linalg.norm(x,axis=1,keepdims=True)```. Softmax is a normalizing function used when your algorithm needs to classify two or more classes. Finding the shape of an array is done by np.shape(array). 

### Vectorizing 
Non-computationally optimal functions can become a huge bottleneck in the algorithm and can result in a model that takes forever to run; vectorization makes the code computationally efficient. Loss is used to evaluate the performance of the model; the bigger the loss is, the more different predictions are from the true value. The following is helpful for vectorized implementation of things:

```python
### VECTORIZED DOT PRODUCT OF VECTORS ###
dot = np.dot(x1,x2)

### VECTORIZED OUTER PRODUCT ###
outer = np.outer(x1,x2)

### VECTORIZED ELEMENTWISE MULTIPLICATION ###
mul = np.multiply(x1,x2)

### VECTORIZED GENERAL DOT PRODUCT ###
dot = np.dot(W,x1)
```
## Logistic Regression with a Neural Network Mindset
Matplotlib is a library used to plot graphs in Python; PIL and Scipy are used to test models with pictures; H5Py is a common package to interact with data stored on an H5 file. We are provided with an H5 file that has data on cats based on the image. First, loading the data is done by the following: ```train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()```. A trick when you want to flatten a matrix X of shape (a,b,c,d) to a matrix X_flatten of shape (b ∗∗ c ∗∗ d, a) is to use: ```X_flatten = X.reshape(X.shape[0], -1).T      # X.T is the transpose of X```. To represent color images, the red, green and blue channels (RGB) must be specified for each pixel, and so the pixel value is actually a vector of three numbers ranging from 0 to 255. One common preprocessing step in machine learning is to center and standardize your dataset, meaning that you substract the mean of the whole numpy array from each example, and then divide each example by the standard deviation of the whole numpy array. But for picture datasets, it is simpler and more convenient and works almost as well to just divide every row of the dataset by 255 (the maximum value of a pixel channel). The main steps for building a Neural Network are:

* Define the model structure (such as number of input features)
* Initialize the model's parameters
* Loop:
* * Calculate current loss (forward propagation)
* * Calculate current gradient (backward propagation)
* * Update parameters (gradient descent)

