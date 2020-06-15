# Introduction
This will contain concepts and coding tools used for the first part of the tutorial. 

## Python Basics with Numpy 
Numpy is used by a large group of people and there are important functions for use with Python. Because in deep learning we mainly use matrices and vectors, we use the numpy library more. Now, implementing the function sigmoid_grad() to compute the gradient of the sigmoid function with respect to its input x. The formula is: $$sigmoid\_derivative(x) = \sigma'(x) = \sigma(x) (1 - \sigma(x))\tag{2}$$
You often code this function in two steps: 
1. Set s to be the sigmoid of x. You might find your sigmoid(x) function useful.
2. Compute $\sigma'(x) = s(1-s)$

This is useful for computing the gradient to optimize loss functions using backpropagation. In deep learning, often np.shape and np.reshape() are used to get the shape of a matrix/vector and to reshape a matrix into some other dimension. Often, when reading a 3D array of shape, you want to unroll or reshape the image to a 1D vector by doing the following: ((length X height X 3),1). You can reshape an array v of shape (a,b,c) into a vector of shape (a*b, c) by doing the following: ```v = v.reshape((v.shape[0]*v.shape[1], v.shape[2]))```. 
