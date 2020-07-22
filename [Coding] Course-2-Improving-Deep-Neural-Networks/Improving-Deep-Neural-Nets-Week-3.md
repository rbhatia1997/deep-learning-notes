# Introduction
This will contain concepts and coding tools used for the second week of the tutorial. 

## Exporting the TensorFlow Library

The first example of using the TensorFlow library that was given to us was for a loss function that simply squares the difference between y and yhat. Writing and running programs in TensorFlow has the following steps:

* Create Tensors (variables) that are not yet executed/evaluated.
* Write operations between those Tensors.
* Initialize your Tensors.
* Create a Session.
* Run the Session. This will run the operations you'd written above.

Therefore, when we created a variable for the loss, we simply defined the loss as a function of other quantities, but did not evaluate its value. To evaluate it, we had to run init=tf.global_variables_initializer(). That initialized the loss variable, and in the last line we were finally able to evaluate the value of loss and print its value. To summarize, remember to initialize your variables, create a session and run the operations inside the session.

A placeholder is an object whose value you can specify only later' you pass values using a feed dictionary. You run the session by running the following: ```sess = tf.Session() and result = sess.run(Y)```. To set up a constant, you do the following: ```tf.constant(np.random.randn(3,1), name = "X")```. To create a placeholder in code, write the following code: ```tf.placeholder(tf.int64, name = 'x')```. You can feed something into placehold by doing the following: ```sess.run(sigmoid,feed_dict = {x: z})```. The great part about TensorFlow is that you can compute loss with just one line of code: ```tf.nn.sigmoid_cross_entropy_with_logits(logits = ...,  labels = ...)```. 

Representing a y vector with columns that are all zeros except one column his is called a "one hot" encoding, because in the converted representation exactly one element of each column is "hot" (meaning set to 1). To do this conversion in numpy, you might have to write a few lines of code. In tensorflow, you can use one line of code:  ```tf.one_hot(labels, depth, axis).```

The next step was involving building a neural network in TensorFlow. The first steps were the following. As usual you flatten the image dataset, then normalize it by dividing by 255. On top of that, you will convert each label to a one-hot vector. These were the steps taken to do that: 

```python
# Flatten the training and test images
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
# Normalize image vectors
X_train = X_train_flatten/255.
X_test = X_test_flatten/255.
# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 6)
Y_test = convert_to_one_hot(Y_test_orig, 6)
```
The first task is creating placeholders for variables. The next step is initializing the parameters. You are going use Xavier Initialization for weights and Zero Initialization for biases. An example is the following: 

```python
W1 = tf.get_variable("W1", [25,12288], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
b1 = tf.get_variable("b1", [25,1], initializer = tf.zeros_initializer())
``` 

Then, we implemented forward propagation in TensorFlow. The following formulas were used: tf.add(...,...) to do an addition, tf.matmul(...,...) to do a matrix multiplication, and tf.nn.relu(...) to apply the ReLU activation. The next step is computing the cost in TensorFlow by using the one line function: tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = ..., labels = ...)). It is important to know that the "logits" and "labels" inputs of tf.nn.softmax_cross_entropy_with_logits are expected to be of shape (number of examples, num_classes). We have thus transposed Z3 and Y for you. Besides, tf.reduce_mean basically does the summation over the examples. This is where you become grateful to programming frameworks. All the backpropagation and the parameters update is taken care of in 1 line of code. It is very easy to incorporate this line in the model.

After you compute the cost function. You will create an "optimizer" object. You have to call this object along with the cost when running the tf.session. When called, it will perform an optimization on the given cost with the chosen method and learning rate.

For instance, for gradient descent the optimizer would be:

optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)
To make the optimization you would do:

_ , c = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
This computes the backpropagation by passing through the tensorflow graph in the reverse order. From cost to inputs. For an ADAM optmiizer, we use the following: ```optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)```.