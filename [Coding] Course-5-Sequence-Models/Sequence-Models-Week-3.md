# Introduction
This will contain concepts and coding tools used for the fifth week of the tutorial. There are three coding tutorials this week.

## Neural Machine Translation 

First we load a dataset that represents human readable dates and their equivalent. Then, we pre-process and map raw text data to the index values. Attention mechanism tells neural machine translation model where it should pay attention at any step. The model involves two seperate LSTMs in this model: pre-attention and post-attention. To create one step of the attention model process, we use a repeater to have s_prev be a certain shape, concatenate a and s_prev, propagate concat through a neural network via densor1 to get e, and then densor2 to propagate e to get the energies variable, then activator to get attention weights, and dotor to get the context vectors to be given to the next LSTM cell. Then, the model runs the input through a Bi-LSTM and calls one-step-attention, generating a y-hat prediction. 

## Trigger Word Detection 

Speech datasets should be close as possible to the application you will want to run it on. You want to detect the word "activate" in working environments; you use positive words (trigger word) and negative words (non trigger word) to create dataset. We will use a spectogram of the audio to easily detect trigger words, which is basically just a fourier transform over a raw audio signal. 

Instead of recording lots of 10 second audio clips with people saying activate, it's easier to record lots of positive and negative words and record background seperately. We just randomly insert activate or negative words in the background audio clip. We use Pydub to manipulate audio and Pydub converts raw audio files into lists of Pydub data structures. When we overlay an activate clip, we update labels for y-t. We train a GRU (gated recurrent unit) to detect when someone has finished saying a word (We will allow the GRU to detect "activate" anywhere within a short time-internal after this moment, so we actually set 50 consecutive values of the label  y⟨t⟩ to 1). Synthesized data is easier to label. 

Network to ingest a spectrogram and output a signal when it detects a trigger word involves a convolutional layer, two GRU layers, and a dense layer. The CONV layer inputs the spectogram and outputs a 1375 step output; two GRU layers read sequence of inputs and dense plus sigmoid layer makes predictions. 