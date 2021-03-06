# Introduction

This Repository contains the notes that I have been writing for the Deep Learning Specialization. It'll cover my assignments and other notes so I can reference them after the course's completion. This is meant for my personal use, but I've made it public in case it's helpful to others. 

## Introduction to Deep Learning

### Neural Networks
Housing prices is a good example of a simple neural network. You input size (x) to a node and that outputs the price of the house (y). The node, a neuron, implements a function (linear approximation, max of zero). A function that goes to zero and then goes to a straight line is called a `ReLU` function - rectified linear unit. If this function is a single neuron; a larger network happens by stacking these neurons together. Instead of predicting a house's price from the size, you can predict it from a variety of factors (e.g. zip codes, bedrooms, etc.). Given input features, a neural network's job is to predict y. Each of the middle circles are called hidden units and take input from the four inputs, allowing the neural network itself to figure out what's important. 

Supervised learning involves having some input, x, and trying to predict some output y. Some examples of this is online advertising and real estate. For image data, you often use CNNs (convolutional neural networks); for time-series data like audio, you use recurrent neural networks (RNNs). Structured data is databases of data (e.g. columns with size, bedrooms, age, ad-id); each of the features has a very defined meaning. Unstructured data refers to data like images, text, or audio that aren't as clear. 

Data collection accumulates a ton of data; scale has been driving neural network progress. Improving scale or increasing amount of labeled data has led to progress. Algorithmic innovation plus the rise of GPUs and such have led to progress. One example is sigmoid functions causing learning to slow due to gradient descent causing parameters to change slowly; therefore, people starting using ReLU functions instead. 

### Basics of Neural Networks - Logistic Regression
You want to process the training set without using an explicit for-loop. Logistic regression is an algorithm for binary classification. An example of binary classification is a cat-detection model that detects the presence or non-existence of a cat. A picture is simply three matrices of 64x64 (RGB); you have to define a feature vector, x, and unroll them into a feature vector. Define a feature vector that lists all the red, green, and blue pixel values (64 x 64 x 3). There is important notation to keep track of when doing machine learning; it's the following:
- (x,y) notation where x is the x-dimensional feature vector and y is the label (0,1). 
- m-training example: {(x,y), (x2, y2), etc.}
- Lowercase m is the number of training examples; m-test is the number of test samples. 
- Matrix (capital X) is defined by taking the training examples and stacking them in columns. 
- Stack Y in columns; Y is a 1 by M matrix. 

#### Logistic Regression
Given x, you want to predict y-hat; y-hat should tell you the chance that something is actually what you say it is (the predicted value). You want to apply a sigmoid function to the quantity (w_x transposed + b), which is just a fancy way of representing a linearity, because you want your y-hat to be between 0 and 1 if you're building a classifier. W is an N_x dimensional vector and b is a real number in this case. Sigmoid function is defined as (1/(1 + e^(-z))). What loss function could we use to see how well our algorithm is doing? The function L, the loss function, will need to convex to find the global minima. The function looks something like: (ylog(y-hat) + (1-y)log(1 - y-hat)). Cost function (J) is used to determine how well y

Now, let's talk about Gradient Descent. We want to find a w,b that makes the cost function as small as possible. Gradient Descent starts at some initial point and takes a step in the steepest downhill direction; after futher iterations, you get to the global minima. We define an alpha, learning rate (controls how big of a step we take on each iteration of gradient descent), that multiplied by derivate of the convex function. The point is that the algorithm is moving towards lowest point. 

Calculus and derivatives are important for understanding machine learning. Derivative means slope of a line; the derivate of the log function is 1/a. Computation graph is helpful when optimizing some special function. You can compute the value of J with a right-to-left pass in a computational graph; computation graph optimizes how you can represent mathematical concepts. One step of backwards propagation on a computation graph yields derivative of a final output variable. Chain rule is an important concept in computation graphs, whereby a->v->J, then if a affects v which affects J, the amount that J changes is the following: (dJ/dV)*(dV/dA). In implementing backpropagation, you will be trying to find some J. In code, when trying to find the derivative with respect to d-var, just write d-var. We end up finding out that the derivative of the loss with respect to z is just (a-y) after doing the chain rule. Now, what if we're trying to do logistic regression on m-examples. Vectorization allows you to remove for-loops in the code; it's become important when dealing with large datasets. Link to the calculus is [here](https://www.coursera.org/learn/neural-networks-deep-learning/discussions/weeks/2/threads/ysF-gYfISSGBfoGHyLkhYg). 

### Python Basics for Deep Learning
Your code should run quickly so you don't have to wait super long. What is vectorization? Well, first, let's talk about how to implement transpose, which is just z = np.(w,x). If you take advantage of non-for loop instructions (e.g. SIMD - single instruction, multiple data), you can use parallelism to make your code run much, much faster. If you want to compute u = Av, you can just do u = np.dot(A,v). If you need to exponentially implement every element; there's a built-in function to compute vectors: u = np.exp(v). You can even do absolute value, maxima, and log without using for-loops. 

When making predictions on the training sample, you just do z(i) = W(transposed)*x_i + b where the activation is a_i = sigma(z_i). We defined a matrix as capital X, which is a (Nx, m) dimensional matrix. First, construct a 1xm matrix: W(transpose) X (1xm). We define some capital Z where you stack lowercase z horizontally. The numpy command is np.dot(W.t, x) + b. In this case, b has to be a real number. Python automatically changes the b variable to a matrix (in a process called broadcasting). For the activation variables, you just stack the lowercase a's and you get a A. Define some dZ variable, which stacks the dz variables, as A - Y. In python, to initiative a sum, you do np.sum. You can just represent db = (1/m) times np.sum(dZ) and dw = (1/m) times dZ (transpose). Therefore, the w is just w - alpha times dw and b is just b - alpha * db. First, create a matrix with the numbers that we have. First, the summation is doing A.sum(axis=0), which sums vertically. Then, the second line of code will simply multiply the matrix by hundred and divide that sum by ```cal.reshape(1,4)```, which shows percentage. Broacasting in Python; if you have an (m,n) matrix and you add/substract/multiply another matrix/number, that matrix goes from a (1,n) to an (m,n) matrix. Beware of rank 1 arrays which don't act like row/column arrays. 

Review of things: 
- A neuron computes a linear function followed by an activation function. 
- You can reshape things to a column vector by adding (,1) at the end of the img.reshape. 
- Broadcasting will change the shape of a matrix when adding. 
- When multiplying matrices, make sure that element-wise multiplication is possible. Number of columns in first matrix needs to equal the number of rows in the second matrix. 
- A np.dot(a, b) has shape (number of rows of a, number of columns of b).

## Shallow Neural Networks

### Overview
Superscript square brackets refer to individual layers whereas superscript round brackets refer to the specific training example. For logistic regression, you have a z followed by an a calculation; for the neural networks, you have multiple z calculations followed by a calculations. The inputs to the neural network are called the input layer. Another layer after that is called the hidden layer; the single node layer at the end is the output layer. Hidden and output layers have parameters associated with them. For logistic regression, you compute some z and then you compute a sigmoid function. The neural network does this a bunch of times. The superscript refers to the layer and the i refers to the node in the layer. We can vectorize the w as a matrix and then add the b vector to represent z as a matrix. 

Activation functions, of which sigmoid functions are a part of, are nonlinear functions that work well (tanh, for example) for gradient descent. Sigmoids are usually used in binary classfication models. Activation models can be different for different layers. Sometimes, when z is large or small, the slope goes close to zero and it slows down gradient descent. ReLU (rectified linear unit) is popular because it bypasses this problem. There's a leaky ReLU function that prevents the slope being zero; it's usually good to just use the ReLU. Non-linear activation functions are important because otherwise the y-hat is just a linear function of the input. No matter how many layers you do, it'd still be a linear function. 

Going over the derivatives of activation functions. For sigmoid activation functions, we find that the derviative is just a*(1-a). For the tanh activation function, the dervaitive is the following: (1-tanh(z))^2. For ReLU, the slope is 0 when z is less than 0 and 1 when z is greater than zero. To train parameters for classification, you need to perform gradient descent. Equations for forward propagation are four-fold for the neural network. Backpropagation step requires you to take derivatives. The dz for derivatives is A - Y; dW is (1/m)(dZ)(A); dB = 1/m * np.sum(dz, axis=1,keepDims=True).

When training neural networks, it's important to initialize things randomly. You can't set things to zero because then every layer your hidden layer is computing the same function (and because the hidden units have same influence on output layer) and are symmetric. You want hidden units to compute different functions. Set w1 to some np.ranodom.randn * 0.01; you can initialize b's to be zero. If weights are too large, when computing activation values, you're screwing with the gradient descent. 

## Deep Neural Networks

### Overview
Deep neural network involve the notation: number of layers is L; lowercase l is the number of units in a certain unit. When you're given just one activation input, x, you compute the activation, which are dependent on previous activation inputs. General rule is that z[l] = W[l] * a[l-1] + b[l]. The activations for that layers are the activation for those functions times z: a[l] = g[l] * z[l]. Demensions of W[l] has to just be n[l] by n[l-1]. For the bias term, b[l], we find that b[l]'s dimensions are n[l] by 1. In addition, dw[l] = n[l] by n[l-1] and db[l] = n[l] by 1. Specifically, having a lot of hidden layers in deep learning, is helpful. Going from low-level, edge features to more high-level, broader features is what deep learning is about. 

For layer l, you have some input a[l-1] that spits out some a[l] using W[l] and b[l]. This outputs a cache that contains z[l]. You use this z for backwards propagation, you're seeing how da[l] impacts da[l-1]. You need W[l], b[l], and z[l] and then get the following: dW[l] and db[l]. You take the input features to get the first layer's activations (caching z[l]). You keep feeding things to the next layers to eventually get the y_hat. For the backpropagation step, you feed da[l] and so on until you go backwards to get dW and Db derivative terms. W and B get updated accordingly. A good thing to know is that for backwards propagation for layer l, for dz[l] = da[l] * g'[l](z[l]) and dW[l] = dz[l] * a[l-1]; db[l] = dz[l] and da[l-1] = W[l]^T * dz[l]. 

Organize hyperparameters and parameters. You need to tell your algorithm the learning rates (alpha), the number of iterations of gradient descent, number of hidden layers, the number of hidden units, choice of activation functions, etc. These are called hyperparameters -> they're parameters that control W and B. See below for the list of equations utilized for the fourth week:

![Math Equations Part 3](photos/photo3.png)






