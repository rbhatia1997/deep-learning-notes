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
Given x, you want to predict y-hat; y-hat should tell you the chance that something is actually what you say it is (the predicted value). You want to apply a sigmoid function to the quantity (w_x transposed + b), which is just a fancy way of representing a linearity, because you want your y-hat to be between 0 and 1 if you're building a classifier. W is an N_x dimensional vector and b is a real number in this case. Sigmoid function is defined as (1/(1 + e^(-z))). What loss function could we use to see how well our algorithm is doing?


