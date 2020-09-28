# Introduction
This will contain concepts and coding tools used for the fifth week of the tutorial. There are three coding tutorials this week. 

## Building Your Recurrent Neural Network - Step by Step

We're going to build an RNN that generates music; we assume that time steps are of size Tx, batches are mini-batches with 20 training examples, and m is the number of training examples. The tensor (nx, m, and Tx) represents input to the RNN. We take a 2D slice for each time step -> using mini-batches of training examples. The activation passed to the RNN from one time step to another is called the hidden state, which is a (na, m, Tx). In order to build an RNN, we implement the calculations needed for one time step of the RNN and then we implement a loop over Tx time-steps in order to process all the inputs, one at a time.  

The first step that we did was implement rnn_cell forward that implemented a single forward step of the RNN cell that takes the current input, a previous state, and outputs an activation via softmax and tanh. A RNN forward pass involves repeating the forward step process. This model that we orginally build is suffering from vanishing gradient problems, so we then go to build LSTM networks. 

LSTMs rely on state/gate changes to keep information. If subject changes its state (singular to plural, for example), the memory of the previous state is outdated so we "forget" that state. Forget gate has a tensor between 0 and 1; if it's close to zero, LSTM forgets the stored state; if it's close to 1, the LSTM remembers the value. Sigmoid used to make gate tensor values range between 0 and 1. Candidate value is a tensor containing info that may be stored in the current cell state; the update gate decides what part of the candidate tensor are passed to the cell state. 

## Dinosaurs Island Code

When gradients are large, they're called exploding gradients. Exploding gradients make the training process more difficult because updates may be so large they overshoot the optimal values during back propagation. There's a helpful function called numpy.clip that you can use. Looks like this: 

```python 
for gradient in [dWax, dWaa, dWya, db, dby]:
        np.clip(gradient, -maxValue, maxValue, out=gradient)
```

Assuming you've trained your model, you're going to want to generate new text characters; the process of generation involves having the network sample one character at a time. You input dummy vector of zeros and run one step of forward propagation, sample, and then update values. 

For RNNs, you forward propagate through the RNN to compute the loss, backward propagate through time to compute the gradients of the loss with respect to the parameters, clip the gradients, and update the parameters using gradient descent. 