# Use Keras to build a English to French translator with various RNN model architecture
## https://medium.com/@patrickhk/use-keras-to-build-a-english-to-french-translator-with-various-rnn-model-architecture-a37439005ae8

This is the second project of my udacity NLP nanodegree and we are required to use Keras as the framework. I have done similar seq2seq model with pytorch before. I find this project quite interesting and is a good chance for me to get familiarize with Keras.<br/>

### Objective
We want to build a RNN model that can input English sentence and output French sentence, so it is actually a translator.

### Dataset
Udacity provided us with two txt files, one containing English sentences and the other containing the corresponding French sentence.<br/>
![p1](https://miro.medium.com/max/700/1*jChPBcR0VfTDinD53z7L7A.png)<br/>
### Data pre processing
Our model cannot read raw text in string directly, we have to first convert them into integer. We have to carry out tokenization to split each sentence into tokens and each token will be encoded with a unique token idx.<br/>
We can write a function to apply Tokenizer from Keras to get the tokens<br/>
```
from keras.preprocessing.text import Tokenizer
def tokenize(x):
    x_tk = Tokenizer()
    x_tk.fit_on_texts(x)
    return x_tk.texts_to_sequences(x), x_tk
 ````
 For example, if our corpus contain just three sentences<br/>
 ```
 text_sentences = [
    'The quick brown fox jumps over the lazy dog .',
    'By Jove , my quick study of lexicography won a prize .',
    'This is a short sentence .']
 ```
 After apply Tokenizer from Keras, each sentence will be encoded into:<br/>

![p2](https://miro.medium.com/max/566/1*1OvkbXYEt3jxtVMqrdBhOg.png)<br/>
The length of sentence in our corpus are varying, some are longer while some shorter. We want to feed our data by batch and within each batch they should have same length. Therefore we will carry out padding to pad or truncate them into equal length.<br/>
We can write a function to get the max length of sentence and pad all sentences to this length with the help of Keras pad_sequences<br/>
```
from keras.preprocessing.sequence import pad_sequences
def pad(x, length=None):
    if length is None:
        length = max([len(sentence) for sentence in x])
        
    return pad_sequences(x, maxlen=length, padding='post')
```
padding = ‘post’ the 0 padding will be added at the end.<br/>

![p3](https://miro.medium.com/max/377/1*jkfvKRWbUhoRzw-sY3EOAw.png)<br/>
Apply the above preprocessing steps in our corpus then we will have English and French sentence in token idx form which are our data set!<br/>
x_train is English token idx and y_train is French token idx. Before moving on to build the RNN model we need to know the Eng and French vocab_size, both can be obtained from len(tokenizer.word_index)<br/>
```
english_vocab_size = len(english_tokenizer.word_index)
french_vocab_size = len(french_tokenizer.word_index)
```
### Write function to map logits back to token label
The output of our model is just probability distribution of class token idx. We have to map the token idx back to its token label. For example if output is 3, we should be able to map it back to token “hello”. Don’t forget to add the pad token in the mapping as well.<br/>
```
def logits_to_text(logits, tokenizer):
idx_to_words = {id: word for word, id in tokenizer.word_index.items()}
    idx_to_words[0] = '<PAD>'
return ' '.join([idx_to_words[prediction] for prediction in np.argmax(logits, 1)])
```
### Build the models and Hyperparameters
I use CuDNNGRU or CuDNNLSTM instead of normal GRU and LSTM because the CuDNN form can train 3–5 times faster on my GPU. However there is no way to add dropout with CuDNNGRU or CuDNNLSTM therefore you should consider whether sacrificing training time for better generalization. For me this project is not a business product to clients therefore I prefer faster training time.<br/>
For the optimizer I follow Udaicty’s suggestion to use SGD, usually I prefer Adam. The learning rate I used was between 0.01 to 0.001, I didnt spend much time on the hyperparameter.<br/>
For the batch_size I use 512, if GPU is out of memory I switch into 256. For high demanding project like in Kaggle, the number of batch_size is important as it will affect the convergence of the model. But in here I just want it to be trained ASAP so I choose the max batch_size allowed for my GTX 1080ti.<br/>

### Model 1: simple RNN model

![p4](https://miro.medium.com/max/647/1*Yt0U1hNdZIMVcsIjjdz4hw.png)<br/>
![p5](https://miro.medium.com/max/700/1*ZTcoxahBbm2NXlm1jyry7g.png)<br/>
validation accuracy is 0.6708 after 5 epochs. (actually I should have trained it 10 epochs to match with other models, so I can have better baseline comparison.)<br/>
### Model 2: Simple RNN with embedding layer
![p6](https://miro.medium.com/max/455/1*ZLK5X0OZP22Gb4UsTMreGw.png)<br/>
![p7](https://miro.medium.com/max/1000/1*DPXswk6wzvs75c923AuNtw.png)<br/>
validation accuracy is 0.9357 after 10 epochs.<br/>
### Model 3: Bidirectional RNN

![p8](https://miro.medium.com/max/692/1*H_WfPaTPN6pRvpnmSohATQ.png)<br/>
![p9](https://miro.medium.com/max/700/1*PBic3ulXgZCautxfyCFZJQ.png)<br/>
validation accuracy is 0.7422 after 10 epochs.<br/>
### Model 4: Encoder-decoder RNN

![p11](https://miro.medium.com/max/700/1*bDT0ohT-5-WqGT130zip0w.png)<br/>
![p12](https://miro.medium.com/max/700/1*SQh7ayZhHXqFCQCGpfHBTQ.png)<br/>
validation accuracy is 0.7021 after 10 epochs<br/>

### Model 5 embedding and bidirectional

![p13](https://miro.medium.com/max/700/1*VltKGpUmTHyV1zW2nDtglg.png)<br/>
![p14](https://miro.medium.com/max/700/1*E1Nq_rCGjcyYRIOH4-4KvA.png)<br/>
validation accuracy is 0.9809 after 20 epochs. (I trained it twice)<br/>
### Testing the model with custom sentence

By just looking at the validation accuracy, it seems the model 5 Bidirectional CuDNNGRU with embedding layer perform the best. So lets try our custom sentence into model 5.<br/>
The custom sentence is “i visit paris in may it was beautiful and relaxing”<br/>
(I really went to Paris for 2 weeks in May and I love it ;) )<br/>
Remember we cannot pass raw text string into our model, we should pre-process it then feed it into our model.<br/>
![p15](https://miro.medium.com/max/700/1*oZK9ZeetksSUiiLrbvfrJA.png)<br/>

The output result is je est paris au mois de mai mais il est relaxant en<br/>
Lets put it into Google translate and see if it can get back into original English sentence:<br/>

![p16](https://miro.medium.com/max/1000/1*FS9Fv8jz2aSpyKTn-9HC-g.png)<br/>
I can get the keyword “May” and “relaxing” but the rest seem..quite bad 😐<br/>
Let’s see if we switch paris into capital letter Paris and try again<br/>

![p17](https://miro.medium.com/max/1000/1*KsHFl5d6ZTgj6bd2DCZ_rA.png)<br/>
Oh it seems much better now 🙂<br/>
Original is “i visit paris in may it was beautiful and relaxing”<br/>
the translated is “I am Paris in May but it is relaxing in”<br/>

### Discussion

1. Overfitting for sure. I use CuDNNGRU to sacrifice generalization for the fast training time, I cannot add dropout in CuDNNGRU. However I do can add dropout in between the dense layer, which should help. Overfitting explains why validation accuracy can reach 0.9809 but translated custom sentence doesn’t seems so accurate
2. Pitiful dataset. Yes the corpus is too small and definitely not enough to train a proper English-French translator. You can see the training log it only took me around 10s to train for one epoch. I guess a “real” dataset may cause me a few hours for one epoch. To solve this problem I should find another dataset
3. The more complicated Encoder-decoder RNN surprisingly performed just slightly better than the simple RNN. I guess for a small dataset, the more complicated architecture tend to underfit more compared to simple architecture. So if our dataset is large, the performance of Encoder-decoder RNN should far better than the simple RNN. Also, usually seq2seq is used in combination with embedding layer for better performance.
4. Embedding layer is the hero. Model adapting embedding layer performed much better.
5. Why use google translate French >English to check if the output French translation is accurate or not? Because I cant read French, I have no other way to verify the accuracy.
-------------------------------------------------------------------------------------------------------------------------------------
### More about me
[[:pencil:My Medium]](https://medium.com/@patrickhk)<br/>
[[:house_with_garden:My Website]](https://www.fiyeroleung.com/)<br/>
[[:space_invader:	My Github]](https://github.com/fiyero)<br/>
