# Introduction
This will contain concepts and coding tools used for the second week of the tutorial. 

## Keras Tutorial 

Unlike TensorFlow, in Keras, you don't need to create a graph and make a seperate sess.run() call to evaluate variables. Keras is great for rapid prototyping. In Keras, you may end up with something like the following:

```python
def HappyModel(input_shape):
    """
    Implementation of the HappyModel.
    
    Arguments:
    input_shape -- shape of the images of the dataset
        (height, width, channels) as a tuple.  
        Note that this does not include the 'batch' as a dimension.
        If you have a batch like 'X_train', 
        then you can provide the input_shape using
        X_train.shape[1:]
    """
    
    ### START CODE HERE ###
    # Feel free to use the suggested outline in the text above to get started, and run through the whole
    # exercise (including the later portions of this notebook) once. The come back also try out other
    # network architectures as well. 
    
    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    X_input = Input(input_shape)

    X = Conv2D(32, (3, 3), strides = (1, 1), name = 'conv0')(X_input)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), name='max_pool')(X)

    X = Conv2D(16, (3, 3), strides = (1, 1), name = 'conv1')(X)
    X = BatchNormalization(axis = 3, name = 'bn1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), name='max_pool2')(X)    
    
    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name='HappyModel')

    ### END CODE HERE ###
    
    return model
``` 

In Keras, after building a model, you train and test the model via four steps. You create the model via the above step; you compile the model by running model.compile; you train the model on the training data via model.fit(); then you test the model on test data by using model.evaluate(). So for creating the model, in this case, the input-shape parameter is a tuple with height, width, and channels - excluding the batch number. Therefore, the input shape is just HappyModel( X_train.shape[1:] ). Then you compile the model via the following (selecting optimizers): ```happyModel.compile(optimizer = "adam",loss = "binary_crossentropy",metrics = ["accuracy"])```. Then, training the model involves the following:  ```happyModel.fit(X_train, Y_train, batch_size=16, epochs=40)```. After doing this training, you can evaluate the model via the following: ```preds = happyModel.evaluate(X_test, Y_test)```. There are other useful tidbits in Keras that may be helpful: ```happyModel.Sumamry()```. There's also the following (plotting): ```plot_model(happyModel, to_file='HappyModel.png')
SVG(model_to_dot(happyModel).create(prog='dot', format='svg'))```. 

## Residual Networks 
This assignment is done in Keras! Deep networks can represent complex functions and learn features at different levels of abstraction. Resnets allow you to shortcut or skip layers; by stacking resnet blocks on top of each other, you can form a very deep network. There are two main types of blocks in a ResNet architecture, which correspond to whether the input/output dimensions are same or different - these are the identify block or the convolution block. THe identity block is the standard blocks used in ResNets and corresponds to the case where the input activation has the same dimension as the output activation. 

The convolution block is the second block type, used when the input and output dimensions do not add up. The difference between this and the identity block is that there's a CONV2D layer in the shortcut path. It's used to resize the input layer to a different dimension. For example, to reduce the size of a dimension's height/width by 2, you can use a 1x1 conv. layer with a stride of 2. 
