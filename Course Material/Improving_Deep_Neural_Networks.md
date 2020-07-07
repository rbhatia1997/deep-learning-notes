# Introduction

This Repository contains the notes that I have been writing for the Deep Learning Specialization. It'll cover my assignments and other notes so I can reference them after the course's completion. This is meant for my personal use, but I've made it public in case it's helpful to others. This is the second course. 

## Week 1 

### Train/Development/Test Sets Introduction

You need to make many decisions when making a neural network. Hyperparameter choice is a highly iterative process. Setting up training/dev/test sets well helps you achieve efficiency. Previously, splitting things to a 70-30% split between train and test was the move; now, the trend is do a 60/20/20 split between data, dev, and test for smaller datasets. We can even do 98% train, 1% train, and 1% dev for huge datasets. For mismatched train/test distrubutions, you need to make sure that the dev and test sets come from the same distribution. 

To try to understand bias and variance, we look at the train set error and the dev set error. If you have low train set error but high dev set error, you may be overfitting to the training set; this means that you have high variance. Where you have high training and dev set error, this could be underfitting the data (high bias). Optimal error or base error is nearly 0%. The bias problem is found by analyzing the training set error; how much the error increases between training and dev set error yields the variance. 

There's a basic recipe for machine learning. Check if the algorithm has high bias - look at the training data performance. Look at a bigger network (train longer), look at a different architecture, etc. Once you reduce bias, you may have a variance problem. Get more data, try regularization, or use a different architecture. Increasing data can reduce variance/doesn't hurt bias and vice versa. 

### Regularizing to your Neural Network

There are things called L1 and L2 regularization that involves adding norms. Sparse means that the w will have a lot of zeros. Lambda in the equation provided is called a regularization parameter and set using the dev set. It's another hyperparameter. In a neural network, you add lambda/2m multiplied the sum of the squared norm. The Frobenius norm of the matrix is utilized to do backpropagation where you add the lambda/m term to the dW updating step. L2 regularization is called weight decay. It's like normal gradient descent, but you multiply by w, which is a numeral less than one. 

![Math Equations Part 5](photos/photo5.png)

So why does regularization prevent overfitting? We are adding an extra term that penalizes the weight matrices from being extra large. Making lambda big, you can zero out or reduce the impact of hidden layers - similar to having a smaller network. You prevent/make it less likely to overfit when your lambda is large, causing w[l] to decrease, cauasing the z to be in a linear range due to the values being small. 

Now, talking about dropout regularization. Go through each of the layer in the network and set some probability of removing a node. Then you remove the ingoing/outgoing nodes and then for each training example, you train on smaller networks. The smaller networks are being trained. What you do is you set a probability of removing the node; then, you create a random matrix that when multiplied by will set that matrix to either 0 or 1. You create some probability the node will be removed and then divide that multiplied matrix (called a3) by the probability of keeping the node. You then run through the steps as normal but knowing that hidden layers are being zeroed out. 

In general, the number of neurons in the previous layer gives us the number of columns of the weight matrix, and the number of neurons in the current layer gives us the number of rows in the weight matrix. Drop out works because you can't rely on any one feature, so you have spread out the weights. This shrinks the squared norm of the weights. You can vary the probability of keeping nodes by layer. Dropout is frequently utilized for computer vision due to the lack of data. Cost function J is no longer well defined due to dropout. 

You can augment your training data to increase the dataset (flipping the dataset horizontally). You can also take random crops of the image. These fake training examples may not be as good as getting new examples, but it's an inexpensive way of synthesizing new data. You can impose random distortions onto things as well. Another technique is early stopping. You plot your dev set error and your training error. You can stop training your neural network halfway -> you get a midsize w, you're overfitting less. The one downside of early stopping is coupling the tasks of optimizing the cost function J and solving overfitting. You can just use L2 regularization instead of early stopping. 

### Setting up the Optimization Problem 

You want the train/test sets to be normalized in the same way. If you use unnormalized features, the range of parameters will be very different; the impact of this is that the learning rates change and it's harder to do gradient descent on a non symmetric graph. Sometimes the derivatives (gradient) can get very big or very small and makes training difficult. The weights W, if they're a little bigger than the identity matrix, can cause the activations to explode; if W is a little less than the identity, the activations will decrease exponentially. These are for very deep neural networks. 

You can set a weight matrix to be np.random.randn(shape) * sqrt(1/n[l-1)]). This is caused Xavier's initialization. You can use ReLU which is the same thing except it becomes 2/n[l-1]. Gradient checking can check if you're doing back propagation is correctly. Using a two-sided difference in derivative computation is more effective and accurate. You'll have some parameters in your parameters; you need to reshape your parameters into a big vector theta. Take the derivative parameters are reshape them into a big vector dtheta. Now, J is a function of the giant parameter vector theta. For each component of theta, take a two sided difference; you end up with a dtheta_approx. You find the eucledian distance between the dtheta_approx and dtheta. 

Use gradient checking only to debug things. If algorithm fails a grad check, look at the components to try to identify the bug. Grad check doesn't work with dropout; but don't forget regularization if that's what you're using. 

## Week 2 

