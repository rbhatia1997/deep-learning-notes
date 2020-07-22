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

### Optimization Algorithms

Having fast optimization algorithms can speed up the efficiency of any team that's doing deep learning. You can use vectorization to process your entire training set; if you just let gradient descent make progress before processing everything. You can split the training set into mini batches. Round brackets refer to the index of a training set; the square brackets is a different layer; curly brackets refer to the mini batches. Batch gradient descent versus mini batch gradient descent. You run a for loop where you implement the steps of gradient descent on a smaller number of elements (epoch is a single pass of a training set). 

Choosing mini batch size is important and is between 1 and the size of the amount of data. Batch gradient descent takes too long; stochaistic gradient descent is inefficient where you lose the speed from vectorization. For small training sets, just use batch gradient descent (less than 2000 examples). Your mini batch size being a product of two is useful; make sure that your X[t] and Y[t] fits in the CPU/GPU memory. 

We can get something even better than these gradient descent algorithms. Exponentially weighted averages are useful. V[t] = B(V[t-1] + (1-B)*(theta(t))). This computes the average of 1/(1-B) days' of data. You can use exponentially weighted averages to understand the data better and account for change. A bigger beta means that it's less susceptible to outlier change as it's demphasizing the impact of recent additions. Doing this in practice is computationally efficient because you're just taking the V-value and overwriting it in memory. 

Bias correction makes the computation of this method more accurate. Because the initial samples will be predicted as lower due to the V0 term being zero, we notice that a better method of implementation is V[t]/(1 - Beta^t). The initial values of vt will be very low which need to be compensated. Make vt=vt1−βt. In essence, it's for when you really care about getting a better estimate early on. 

Gradient descent with momentum works faster than the standard gradient descent algorithm. You compute an exponentially weighted average of the gradients, and then use that gradient instead to update the weights. We compute what Vdw and Vdb are based on the formula for the weighted average. Allows algorithm to mainly ignore changes in the vertical direction, but push towards the direction of the minimum during gradient descent. 

![Math Equations Part 9](photos/photo9.png)

Another algorithm can speed up gradient descent, which is RMSprop or root mean square prop. You want to slow learning in the b direction and speed up learning in the w direction. Basically the same as above but you're dividing by a square root term to dampen the oscillations more. One nice thing about this is you can increase the learning rate then. 

The Adam optimization algorithm involves implementing RMSprop and momentum and putting them together. We tune the hyperparameter alpha, we keep a hyperparameter of B1 and B2 (0.9 and 0.999). For epsilon, we use 10^-8. Adam stands for adaptive moment estimation where B1 is the mean of the derivations and B2 is the exponetially weighted averages of the squares. 

Another thing to speed up the algorithm is slowing down the learning rate over time. This is learning rate decay. As your alpha gets smaller, your steps would oscillate around a tighter region as it approaches the minimum. Set the rate to 1/(1+parameter*epoch-number). Some people also set alpha to some exponential or some square-root constant. Some people do manual decay and manually decrease alpha. 

In high dimesional spaces, you're more likely to run into a saddle point than local optimum. This is where the derivative is zero. If local optimum aren't a problem, then plateaus are a problem (where zero is the derivative for a while). It takes a while to get out of the plateau. Momentum, RMSProp, and ADAM are useful to combat this.

## Week 3 

### Hyperparameter Tuning

There are a lot of hyperparameters. Some parameters are most important than others. Alpha is very important; momentum term is important; mini-batch and hidden units are important. Layers/learning rate decay are also good considerations. Choose points at random and try out the hyperparameters on the randomly chosen points. Using a coarse to fine scheme; you might zoom into to a smaller region and sample within that region randomly. 

Search for hyperparameters on a log scale. You sample uniformly on this log scale; computing hyperparameters for exponentially weighted averages is also unique. You should sample values of 1-beta; you sample values of beta for 10^-1 to 10^-3. When beta is close to 1, sensitivity of the results change. Causes you to sample more densely. 

Different machine learnings applications may not work in different communities. Retest your hyperparameters. Babysitting a model is one method where you constantly change the model as it trains over many days; this is what happens when you don't have much computational power. You can also train many models in parallel. 

### Batch Normalization

Makes hyperparameter search much easier. Will let you train your neural networks much easier. Normalizing input features can make the learning process much faster. Can you normalize activation layers/hidden layers to train w3/b3 faster? Batch normalization involves normalizing z2. Given some intermediate values in the neural net, compute the mean and then you compute the variance. You normalize the z[i]'s. We introduce gamma and Beta (learnable parameters) that we use in later computations in the neural network. Normalizing input features can speed things up; batch normalization can normalize the mean/variance of some of the hidden unit values based on certain parameters. 

Usually batch norm is applied on mini-batches. You just compute variance on the minibatch. You iterate on the number of minibatches; you do forward prop (use batch norm to replace z[l] with z tilda [l]). You use back propgation to get dW, DB, etc. and update the parameters. You can then use gradient descent with momentum/RMSProp/etc. 

Makes weights more robust to changes deeper in the network rather than initially. Covariate shift is retraining the algorithm if the x changes; need to retrain becomes worth if the ground truth function shifts. Batch norm ensures that normalization occurs; limits the amount which updating amounts in early layers can impact the later layers. Batch norm means the early layers doesn't dictate as much; layers are more independent. Each mini batch is scaled by the mean/variance computed on that mini batch; adds noise to values within the minibatch. Slight regularization effect. 

During test time, you may not have the minibatch data. You come up with a seperate mu and sigma squared by coming up with an exponentially weighted average. 

### Multi-class Classification

What if we have many classes? What if you want to recognize many other classes? You want to know the proability of being in a certain class. Softmax layers allows you to do this. You compute the linear part; now you compute the softmax activation function. You do e^z[l]; the output is the vector t normalized which gives you percentages for multiple classes. This process is called the softmax activation function. 

Training a softmax classifier. Name comes from opposite of hardmax, which looks at the element of z, which makes the biggest vector 1 and everything else zero - softmax is a softer version of this. Softmax reduces to linear regression when there are only 2 classes. 

### Introduction to Programming Frameworks

You're going to use software/deep learning frameworks. It's more efficient to do things using deep frameworks. Ease of programming, running speed, and open source code are how you determine whether a deep learning framework is good or not. 

This week's exercise will take some time. You can use TensorFlow to find values to minimize the cost function. You import numpy and import tensorflow as tf. You use tf.Variable to define variables. You can train using a function called ```tf.train.GradientDescentOptimizer```. You initialize variables and then you run a session. You can get training data in a TensorFlow program by doing tf.placeholder(tf.float32, [3,1]). You provide the values for this later in the training step where you write feed_dict={x:coefficients}. The heart of a TensorFlow program is to compute a cost and then figure out how to minimize that cost. 












