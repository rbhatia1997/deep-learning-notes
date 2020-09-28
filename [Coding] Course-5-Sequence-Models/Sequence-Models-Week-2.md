# Introduction
This will contain concepts and coding tools used for the fifth week of the tutorial. There are three coding tutorials this week.

## Debiasing - Operations on Word Vectors
We're using fifty dimensional GloVe vectors to represent words. This loads words, a set of words in the vocabulary, and word_to_vec_map which dictionary maps words to their GloVe vector representation. To measure similarity between two words, we need a way to measure the degree of similarity between two embedding vectors for two words. This is the dot product over the norm's multiplied. If u and v are similar, their cosine similarity are close to 1; if not, it takes a smaller value. In other words,     cosine_similarity = dot/(norm_u*norm_v). 

Word analogy involves trying to find a word D such that the associated word vectors are related in the following manner:  $e_b - e_a \approx e_d - e_c$. 

## Emojify

In this case, we're building something that selects emojis for a given input text. Word vectors allows training datasets to be more flexible. We started with a basic model EMOJISET that contains 127 sentences. In the code, we implement sentence_to_avg() which converts sentences to lowercase and then splits it into a list of words; then, for each word, we give the GloVe representation and take the average. Then, after getting the average, we pass the average value through forward propagation, compute the cost, and then backpropagate to update the softmax parameters. 

In the Emoji V2 case, we want to use Keras mini batches. The common solution to handling sequences of varying length is to pad. Finally, creating a pretrained embedding layer and building the model. 
