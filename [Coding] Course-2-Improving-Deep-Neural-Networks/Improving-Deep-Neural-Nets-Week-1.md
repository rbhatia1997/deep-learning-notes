# Introduction
This will contain concepts and coding tools used for the first week of the tutorial. 

## Initialization
Training your neural network requires specifying an initial value of the weights. A well chosen initialization method will help learning. Well chosen implementation assists with speeding the convergence of gradient descent and increase the odds of gradient descent coverging to a lower training/generalization error. 

There are two types of parameters to initialize in a neural network. One is the weight matrices and the other is the bias vectors. We initialize by changing ```parameters['W' + str(l)] = None``` and ```parameters['b' + str(l)] = None```. A reminder here that:  WL is weight matrix of shape (layers_dims[L], layers_dims[L-1]) and bL -- bias vector of shape (layers_dims[L], 1). In general, initializing all the weights to zero results in the network failing to break symmetry. This means that every neuron in each layer will learn the same thing, and you might as well be training a neural network with  n[l]=1n[l]=1  for every layer, and the network is no more powerful than a linear classifier such as logistic regression. The weights $W^{[l]}$ should be initialized randomly to break symmetry. 

Next, we were told to do initialize the weights to large, random values. This is done by basically using ```np.random.randn(..,..) * 10 for weights```. Initializing weights to very large random values does not work well. Hopefully intializing with small random values does better. The important question is: how small should be these random values be? We use He implementation for this part. Xavier initialization uses a scaling factor for the weights  W[l]  of sqrt(1./layers_dims[l-1]) where He initialization would use sqrt(2./layers_dims[l-1]). This scaling factor is the same as the previous implementation except instead of multiplying by 10, we multiply by this extra term! 

## Regularization 
We need to prevent overfitting as well, especially when we don't have a ton of test data. The standard way to avoid overfitting is called L2 regularization, which consists of modifying the cost function to include something called the L2 regularization cost (in addition to the cross-entropy cost). This looks like the following: 

![Math Equations Part 6](photos/photo6.png)

In practice, this looked like multiplying ```(lambd/(2*m))``` by np.sum(np.square(W-terms()). Of course, because you changed the cost, you have to change backward propagation as well! All the gradients have to be computed with respect to this new cost. For reach dW term, you need to add the regularization term's gradient: (λ/m)*W.The value of  λλ  is a hyperparameter that you can tune using a dev set.
L2 regularization makes your decision boundary smoother. If  λ is too large, it is also possible to "oversmooth", resulting in a model with high bias. 

In regards to regularization, dropout is a method that can be used. In dropout, when you shut some neurons down, you actually modify your model. The idea behind drop-out is that at each iteration, you train a different model that uses only a subset of your neurons. With dropout, your neurons thus become less sensitive to the activation of one other specific neuron, because that other neuron might be shut down at any time. We create a variable d[1]  with the same shape as  a[1] using np.random.rand() to randomly get numbers between 0 and 1. You use a vectorized implementation, so create a random matrix  D[1]=[d[1](1)d[1](2)...d[1](m)]D[1]=[d[1](1)d[1](2)...d[1](m)]  of the same dimension as  A[1]. ```X = (X < keep_prob).astype(int)``` is a python one-liner for an if-else statement. Set  A[1] to  A[1]∗D[1]A[1]∗D[1] (You are shutting down some neurons). You can think of  D[1] as a mask, so that when it is multiplied with another matrix, it shuts down some of the values. Divide  A[1]A[1]  by keep_prob. By doing this you are assuring that the result of the cost will still have the same expected value as without drop-out. (This technique is also called inverted dropout).

Now, you need to implement backwards propagation with dropout. You had previously shut down some neurons during forward propagation, by applying a mask  D[1]  to A1. In backpropagation, you will have to shut down the same neurons, by reapplying the same mask  D[1] to dA1. During forward propagation, you had divided A1 by keep_prob. In backpropagation, you'll therefore have to divide dA1 by keep_prob again (the calculus interpretation is that if  A[1]  is scaled by keep_prob, then its derivative  dA[1]dA[1]  is also scaled by the same keep_prob).

## Gradient Checking 

You can use code for computing the cost to verify the code for computing dj/dtheta, which is what is computed during back propagation. The following is true for 1D examples and then for N-dimensional examples. Gradient checking verifies closeness between the gradients from backpropagation and the numerical approximation of the gradient (computed using forward propagation). Gradient checking is slow, so we don't run it in every iteration of training. You would usually run it only to make sure your code is correct, then turn it off and use backprop for the actual learning process.

![Math Equations Part 7](photos/photo7.png)

![Math Equations Part 8](photos/photo8.png)