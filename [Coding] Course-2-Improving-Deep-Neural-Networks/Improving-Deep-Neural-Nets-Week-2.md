# Introduction
This will contain concepts and coding tools used for the second week of the tutorial. 

## Optimization Methods 

We start the tutorial by working on implementing batch gradient descent, which involves simply taking the parameter and subtracting the learning rate * the derivative from it. Now we implement stochastic gradient descent, which is equivalent to mini batch gradient descent where each mini batch has one example. The update rule from above doesn't change but you would just implement it a bit differently. Implementing SGD uses three for-loops, which is over the number of iterations, number of training examples, and over the layers. Mini batch gradient descent is a much better approach to take. You need to tune a hyperparameter learning rate a; a well tuned mini batch size outperforms gradient and stochastic gradient descent. 

To get a mini batch, you either create a shuffled version of the training set or you partition the shuffled (X,Y) into mini batches. So in our case, we shuffle the (X,Y) using permutation command. Then, we partition the shuffled X and Y. Usually, the mini batch size is some power of two. 

Because mini-batch gradient descent makes a parameter update after just seeing a subset of examples, the direction of the update has variance; momentum can mitigate oscillation when reaching convergence. To update the parameters with momentum, we use the following momentum update rule: 

![Math Equations Part 10](deep-learning-notes/photos/photo10.png)

The larger the momentum  β  is, the smoother the update because the more we take the past gradients into account. But if  β is too big, it could also smooth out the updates too much. Common values for beta range from 0.8 to 0.999. If you don't feel inclined to tune this,  β=0.9  is often a reasonable default.Tuning the optimal β for your model might need trying several values to see what works best in term of reducing the value of the cost function  J. Momentum takes past gradients into account to smooth out the steps of gradient descent. It can be applied with batch gradient descent, mini-batch gradient descent or stochastic gradient descent.

Adam is one of the most effective optimization algorithms for training neural networks because it combines RMSProp and Momentum. To implement Adam, we do the following: 

![Math Equations Part 11](deep-learning-notes/photos/photo11.png)

There's a super useful function called ```np.zeros_like``` that I use to initialize arrays of zeros. Afer that, I implemented the functions above. 




