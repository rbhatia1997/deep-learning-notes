# Introduction
This will contain concepts and coding tools used for the third week of the tutorial. 

## Neural Style Transfer

We merge a content image and a style image to create a generated image, which gets the style. NST uses a pre-trained CONV network and builds on top of that -> this is transfer learning. We load this VGG-19 model via load_vgg_model -> stores in a python dictionary. To run an image through this network, you feed an image to the model via tf.assign. To activate a certain layer you run a TF session on a tensor specifically. We would like the generated image to have simlar content; we choose a layer in the middle of the network. We forward propagate image C and image G and then we define some content cost function. To retrieve dimensions from a tensor X, use: X.get_shape().as_list(). If you prefer to re-order the dimensions, you can use tf.transpose(tensor, perm), where perm is a list of integers containing the original index of the dimensions. For example, tf.transpose(a_C, perm=[0,3,1,2]) changes the dimensions from  (m,nH,nW,nC)(m,nH,nW,nC)  to  (m,nC,nH,nW)(m,nC,nH,nW). We convert to tensor, then we unroll/reshape, and then compute the cost. Content cost takes hidden layer activation and measures how different the activations are; this checks similiarity between G and C. 

Now, computing the style cost. You can define style cost function as a set of vectors as a matrix of dot products; checks similiarity of certain vectors and we compute style matrix by multiplying unrolled filter matrix with its transpose. The style matrix or Gram matrix is then (GA = tf.matmul( A, tf.transpose(A)). Then you can throw this into the function that computes the style cost. You get better results if you merge style costs from several different layers. Each layer gets lambda weights that reflect how much they contribute to style. For each layer:
* Select the activation (the output tensor) of the current layer.
* Get the style of the style image "S" from the current layer.
* Get the style of the generated image "G" from the current layer.
* Compute the "style cost" for the current layer
* Add the weighted style cost to the overall style cost (J_style). 

Now actually running the thing; we compute our total cost via that equation. Start the interactive session, load/reshape images, and then initializae the generated image, load the pretrained model, call the total cost function after computing content and style cost,  and using ADAM optimizer and implementing the model. 

## Coursera Face Recognition 

We'll use a pretrained model that represents a ConvNet activations via channels first convention used in lecture/previous programming assignments. By encoding an image, an element wise comparison produces a more accurate judgment. Because FaceNet takes a ton of time to train, we can load weights that someone else has already trained. Bsaically we use a 128-neuron fully connected layer as the last layer and then we use the encodings to compare two faces via the distance formula. 

We're using triplet loss. We then compute the loss and load the pre-trained model. When applying the model, we wanted to build a face verification system. We can build a database containing one encoding vector for each person allowed to enter the office (we run the following code to build the database) and the database maps the person's name to a 128 dimensional encoding of their face. 
