# Introduction
This will contain concepts and coding tools used for the third week of the tutorial. 

## Building Deep Neural Networks 

The steps we used for implementing this is to intialize the parameteres, do a ReLu forward, do a ReLu backwards, and update parameters and compute losses in order to make more and more accurate predictions. The model's structure here is to go from linear to ReLu to linear to sigmoid. To randomly initialize the weight matrices, we utilize ```np.random.randn(shape)*0.01``` with the correct shape. The biases of course can be implement with zeros via np.zeros(shape). The initialization for a deeper L-layer neural network is more complicated because there are many more weight matrices and bias vectors. When completing the initialize_parameters_deep, you should make sure that your dimensions match between each layer. We defined some linear_forward function that basically just computes ```np.dot(W,A) + b```. From this, we can build our linear-activation forward functions; for even more convenience when implementing the  L-layer Neural Net, you will need a function that replicates the previous one (linear_activation_forward with RELU) L−1  times, then follows that with one linear_activation_forward with SIGMOID. See the following photo for how to implement things for backwards propagation. 

![Math Equations Part 4](photos/photo4.png)

After this, we create a function that merges the helper functions to compute the linear activation function for backwards propagation. If g is the activation function, then sigmoid_backwards and relu_backwards computes dZ[l] = dA[l]*g'(Z[l]). Now, to compute backpropagation for the whole network, you use cached variables and iterate through the hidden layers backwards starting from layer L. We know the ouptut of A[l] is sigmoid(Z[l]) and therefore dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)) # derivative of cost with respect to AL. We create a grads dictionary to store each dA, dW, and dB value. The last step is updating parameters using gradient descent (W[l] = W[l] - alpha * dW[l] for example). 

## Deep Neural Network Application

First, we reshape the training and test examples after loading them. 

```python
train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.
```
For the exercise, we looked at 2-layer neural networks and L-layer deep neural networks. For the two-layer neural network, we first initialize parameters, perform the linera activation forward (on the RELU and Sigmoid function), compute the cost, perform the backwards linear activation, and then update parameters. For performing the L-layer neural network, you do the linear->Relu->linear->sigmoid. This process starts by initializing parameters, doing an L_model_forward, computing the cost, performing the L_model_backwards, and then updating parameters. 