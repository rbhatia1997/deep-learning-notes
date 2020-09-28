# Introduction

This Repository contains the notes that I have been writing for the Deep Learning Specialization. It'll cover my assignments and other notes so I can reference them after the course's completion. This is meant for my personal use, but I've made it public in case it's helpful to others. This is the fifth course.

## Week 1 

### Sequence Models

Sequence models involve sequence data, where the data is a sequence of data over time (e.g. DNA/sentiment classifaction). Named entity recognition is used by search engines to identify proper nouns in a sentence. You can consider a sentence as having features (words) and you use bracket superscripts as the notation for which word we're at. Also, t_x is the notation for length of input sequence; t_y is the length of the output sequence.

Words in a sentence; the first thing to do is to create a vocabulary (dictionary). This could be the top 10,000 words. You can use one-hot notation to indicate this (just put a 1 where you have an index in the dictionary). If there is a word not in the vocabulary, you create an unknown word vector. You could use a standard neural network to try to determine which word is in the sentence; the problems are the fact that each input/output have different lengths. The second problem is that the naive architecture doesn't share features. We want to strive to reduce paramters/streamline our network. 

A recurrant neural network involves getting information from the previous time steps and utilizes that information to make predictions. The activations are used in each step of the way. The weakness for RNN is that it only utilizes information from the words/features previously [at least, this is for the unidirectional RNNs]. The forward propagation situation involves using some tanh or ReLU and the sigmoid/softmax will be used for output. 

In forward prop, you use your input sequence to get the activations at the time steps, which rely on shared parameters. Backpropagation needs a loss function, which would be element-wise loss - for a certain word, output some probability (cross entropy loss). You compute loss for all the timesteps. There are times where input/output length can be different lengths. You can modify the RNN architecture to handle these cases. Many to many architecture means many inputs, many outputs; many to one means many inputs, one output. You can even have a one-to-many output where your input is some x and your RNN output is a sequence (like musical notes). For translations (an example of many-to-many architecture), you have an enoder to read the sentence and a decoder that outputs the sentence to another example. 

Language models yields the probability of how likely something is; given a sentence, what's the probability of a particular sentence. You need a training set/corpus of english text for language modeling. You build a vocabulary and tokenize (convert to one-hot form) the sentences (adding an EOS token, or end of sentence token). UNK token means unknown word. At each timestep, you use a softmax to see what the probability of each word is; you also give the next timestep the previously predicted word. Predicts one word at a time going left to right; overall loss is the sum of the loss of the individual steps. 

If you want to sample sequence from trained RNN, you can randomly select (np's random choice) from the predicted y-hat outputs; you pass those outputs to the next step in the RNN. You can also build character level RNN where your outputs are characters; doesn't require you to worry about unknown word tokens but you have much longer sequences. Word level language models are being utilized more than these. 

Vanishing gradient problem happens in RNNs. When you carry out forward prop, the gradient from output y has a very hard time (in deeper networks) impacting the layers previously. May be hard for neural networks to generalize what tenses should be utilized for example due to this problem. This is a weakness of the RNN algorithm. Gradient clipping involves clipping your gradients so they don't explode and impact your parameters. 

### Gated Recurrent Unit, LSTM, Deep RNNs, Bidirectional RNNs

Gated Recurrent Unit is a modification to the RNN hidden layer and helps a lot with the vanishing gradient issue. So in this GRU case, we have a memory cell (C) that provides memory to tell whether cat is singular or plural. The GRU will output an activation value equal to this C. We'll have a gate (gamma u), a value between 0 and 1 [always 0 or 1 - in practicality, it's a sigmoid function]. The C value is a 0 or 1 depending on singular or plural; the gate decides when to update the value of C. The full GRU involves a gate that computes the relevancy of opening a gate in relation to the other time steps.

The LSTM unit also helps you do what a GRU does but much better. It's a more general version; we don't have the case that a_t = c_t; you don't use a relevance gate. If you connect the LSTMs together, there's a line at the top that basically lets you keep a value the same across deep layers. Peephole connection is when the gate values depend on previous memory cell values as well. LSTMs actually came up earlier than GRU; GRU is simplier so more people utilize that despite the LSTM being more powerful. 

You need more information in a sentence than just previous words; bidirectional RNNs involve a forward recurring component and a backward component. This network defines an acyclic graph. Bidrectional with LSTM blocks are usually utilized. The disavantage is that you need the entire data -> you need to wait for someone to finish their speech before processing it. You can stack multiple layers of RNNs together to build deeper models. 

## Week 2

### Introduction to Word Embeddings

One-hot representation doesn't let you generalize across words. It doesn't know the relationship between different words. There's a featurized representation - a set of features or values for each of the words. You rank words based on how well they match the features. If you use this representation, the representations for similar words take on similar values; we can generalize better across similar words. Benefits of this method is the fact that generalization can happen even with a small training set. You can do transfer learning with large text corpus or download pre-trained embeddings online. People use embedding and encoding in similar ways; encoding is fixed for each of the words in the vocabulary. If you subtract some of the vector embeddings, you can try to find a vector/word such that it's close in meaning; find some word so that e_man - e_women ~ e_king - e_w. You want to maximize the similiarity between e_w and e_king - e_man + e_woman. Cosine similiarity is how you can determine similarity between things; square similarity is another thing. We learn an embedding matrix E (300 x 10k words matrix); columns is the different matrix embeddings. In practice, you use a specialized function to look up stuff rather than use a matrix vector multiplication due to speed. 

### Learning Word Embeddings

First, if you want to learn what the last word of a sentence might be, you multiply the one-hot vector by the embedding vector. You feed all the embedding vectors to a softmax layer that outputs to the dictionary. What's more commonly done is that you look at a four-word history rather than all embeddings and then have the softmax predict the output. You use the same matrix E for all the words; you use backprop to get gradient descent to find what the next word is. You can choose context (e.g. four words on the left and right). 

Let's say we have 10k words; we want to learn mapping from some context C to some target T; you can start with a one-hot vector feed it to a softmax unit. Loss will be the yi log yi-hat. By skipping words from left and right, you can try to evaluate a set of embedding vectors. However, the cost is computational speed as you're evaluating a sum over all the words in the vocabulary. Using a hierarchical softmax classifier is a solution; you can do a binary classification model that classifies (tree of classifiers). Usually this is used so that common words are more at the top versus less frequent words. Sampling conext c? That can work randomly. Distribution is taken not entirely randomly but rather takes into consideration how frequent the words are showing up. 

So let's say we pick a context word and then created a dataset that has a target value of 1 when the context and word makes sense and then 0 for context and word (from randomly selected words). You train binary classifiers on the negative samples - this method called negative sampling. You can sample to how often things appear (don't do this); take the two extremes and sample proportion to the words to the power of 3/4ths. This helps you learn word vectors. 

GloVe algorithm is pretty popular. Global vectors for word representation is the meaning. Idea is to solve for theta and e such that we learn vectors so the end product is a predictor for how often two words appear together.

### Applications with Word Embeddings 

Sentiment classification is hard in that you may not have a lot of training data. Sentiment may be a star rating system. You could utilize a sentiment classifier to see how positive or negative a comment is. One challenge is you may not have a huge training dataset. You take words, look up the one hot vector, extract the embedding vector for everything; you can take the vectors and then just sum or average them, which gives you a 300 dimensional feature vector and then use a softmax to get a star rating. It averages the meaning of all the words. It ignores word order, however, so a negative review that uses "good" a lot may misclassify. RNN for sentiment classifier; you find one-hot vector, multiply by embedding matrixj, and then you can feed into an RNN -> you can compute a representation in the last time step. 

Debiasing word embeddings is important. We find clear examples of bias. ML algorithms make very important decisions, so we need to make sure that we need to diminish bias/eliminate undesirable bias. We identify the direction of a specific bias by taking the embedding vector for he/she and take a few of them and average them. Bias direction is a 1D subspace (can be a higher dimensional, can also be utilized to SVU). Next, you build a neutralization step where for every non-defintional word, you project to get rid of bias. Then, we do an equalization step. 

## Week 3

### Various Sequence to Sequence Architectures 

We create an encoder network which feeds input (sequence) of words; the RNN takes the encoding output and can be trained to output a translation of those words. Decoder network is used to give the translation. Captions also utilize this idea. You can have a pre-trained network to be an encoder output; you can feed it to an RNN who can generate a caption for images one word at a time. 

Language model indicates the probability of a sentence. Machine translation starts with an encoding network that's intitialized with the representation of the input sentence. It models the probability of the output of the translation based on an input sentence. This is called a conditional language model. You don't want to sample outputs at random here; you want to find a english sentence that maximizes a translation. Beam search is the algorithm used. Greedy search is just picking the most likely algorithm (best first word and then the best second word) doesn't work. 

Beam search will try to pick the first word in the sentence by setting a beam width (how many words to evaluate) and store that information in memory. The next step is to consider what should be the second word for these words. By hardwiring the input, you can use the network fragment to illustrate what pairs are most likely. You evaluate all the options and pick the top three possibilities. You make copies of the network depending on the beam width. Length normalization is something you can do to improve the algorithm. You use logs to max (P(y|x)) to reduce rounding error in the algorithm; for long sentences, we have smaller probabilities (shorter translations are bias towards those). We then add some normalization term for the length of the terms. Choosing beam width -> very large beam width has better result but it's slower; very small beam width is a worse result but you get a faster result. In practice, people usually use beam width around 10. 

Beam Search doesn't always output the most likely result due to its speed. The RNN computes P(y|x) and the most useful thing is to compute P of y-hat or P of y*; depending if this is true or not, you can tell if your RNN or your beam search algorithm is not working. If P of y* is larger than P of y-hat, beam search is failing because it's supposed to find a larger value; if P of y* is smaller than P of y-hat, then the RNN is at fault. Through this process, you can see what fractions are due to beam search versus the RNN. BLEU helps developers evaluate machine translation. 

The Attention model/algorithm addresses the problem of long sentences. This model looks at parts of sentences. Attention model computes attention weights (how much importance does this word have to the translation). The state and activations at a time influence how much attention is given to certain words. Using context vectors, we generate output one word at a time. It takes quadratic cost to run the attention model. 

### Speech Recognition - Audio Data 

Speech recognition is to take audio clip and get a audio transcript. A common pre-processing step is to plot spectograms and pass that into algorithms. Phenomes aren't necessary; you could build an attention model for speech recognition. However, people use CTC (connectionist temporal classification); collapse repeated characters not seperated by "blank." Trigger word detection are Amazon Echo, Baidu DuerOS, Siri, and Google Home ("Hey Siri). You can make your computer do something by activating a trigger word. You can train it on data where you set a label to be 1 for the label but 0 for everything else (this creates an imbalanced training set). You could make it output 1 for a duration of time instead of once but it's a bit of a hack. 