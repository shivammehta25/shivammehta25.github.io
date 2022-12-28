---
layout: post
title: A basic Markov model for NLP
date: 2019-03-27 19:44 +0100
categories: [machine-learning]
tags: [linear-algebra , machine-learning, markov-model, math, nlp]
image:
  path: thumbnail.png
  alt: A simple markov chain
math: true
---
## (Migrated from old blog)

Alrighty,

So This is something technical. Let's talk about Markov chain theory. What is it? it sounds very technical xD Okay. Let me hit you with Wikipedia Hard definition. Brace yourself.

> A **Markov chain** is a [stochastic model](https://en.wikipedia.org/wiki/Stochastic_model) describing a [sequence](https://en.wikipedia.org/wiki/Sequence) of possible events in which the probability of each event depends only on the state attained in the previous event.
>
> Someone Genius in Wikipedia - [https://en.wikipedia.org/wiki/Markov\_chain](https://en.wikipedia.org/wiki/Markov_chain)

So well for me, I like to put it this way that the, it is a transition state diagram where the cost to move from one state to another is the probability of that event happening. Like if you see in this image we will get that.

![What to eat markov model](thumbnail.png)

Markov Model

Lets talk about food. Ofcourse we all love it :P 😍😛
So If I eat pasta today the probability of me eating pasta tomorrow is 0.50 and the probability of me eating burrito tomorrow is 0.30 and the probability of me eating Dahl (I have no idea what it is, it shoudl be 0.0 but for the sake of mathematics lets assume) it is 0.20. So we can create a transition state matrix of

$$ P = \Bigg[ \begin{array}{c c c} 0.50 & 0.30 & 0.20 \\ 0.25 & 0.50 & 0.25 \\ 0.40 & 0.40 & 0.20 \end{array} \Bigg] $$

So let the starting state $S_0$ be equal to $ S_0 = [ 0.4 \\ 0.5 \\ 0.1] $ so we can determine the probability of going to next state by simply $ \langle S_0, P \rangle$

$$ S_1 = [0.4  0.5  0.1 ] \cdot \Bigg[ \begin{array}{c c c} 0.50 & 0.30 & 0.20 \\ 0.25 & 0.50 & 0.25 \\ 0.40 & 0.40 & 0.20 \end{array} \Bigg] = [0.365 \  0.41 \  0.225 ] $$

Okay, too much of Mathematics lets switch to code ( now we are talking) . So we have a text dataset of lots of corpus given by my teacher. Click to download, we need it for furthur coding.

[whatrocks-datasets-commencement-7](whatrocks-datasets-commencement-7.tar)

So let's begin with basics and load this textual dataset into out python. I am coding side by side while writing lets see where we end upto.

```python
import glob
from collections import defaultdict
import nltk
import random
```

So glob is used to do file and directory listings and more features like this it is an alternative to os.listdir() and isFile functionality ( in this sense). I like to put a more object-oriented approach and design in my code, just provides me accessibility and scalability and I was trained well by Nilofar to use OOP paradigms. (If she sees it she will be very proud) So I will convert mostly everything into classes and objects instead of a straight forward functional code.

Given that I will like to create a class, let's call it TextGenerator and have an init function which just has, for now, a dataset variable, and we will be populating it using a function that will read all files and align the next pair of words into a key value dictionary (similar to a bi-gram model).

```python
class TextGenerator:
    def __init__(self):
        self.dataset = defaultdict(list)

    def generate_dataset(self,file_pattern='datatask4/*.txt'):
        '''
        Generates Dataset from the nearby txt files
        :return: None
        '''

        for filename in glob.glob(file_pattern):
            with open(filename, 'r', encoding="utf-8") as file:
                file_data = file.readlines()
                file_words = nltk.word_tokenize(''.join(file_data))
                for i in range(len(file_words) -1):
                    self.dataset[file_words[i]].append(file_words[i + 1])
        self.dataset = dict(self.dataset)
```

Now for us to use it we need to create an Object and call the specific function! Cool enough

```python
text = TextGenerator()
text.generate_dataset()
```

Now the text.dataset has key-value pairs of bigrams like these

```text
{'In': ['this', 'its', 'so', 'the', 'like', 'that', 'this', 'how', 'a', 'the', 'the', 'the', 'the', 'such'], 'this': ['refulgent', 'world', 'mind', 'way', 'homely', 'sentiment', 'element', 'rapid', 'same', 'law', 'sentiment', 'sentiment', 'sentiment', 'piety', 'world', 'infusion', 'truth', ',', 'primary', 'faith', 'perversion', 'occasion', 'jubilee', 'high', 'daily', 'point', 'eastern', 'charm', ',', 'excellency', 'holy', 'occasion', 'ill-suppressed', 'moaning', ',', 'thoughtless', 'docility', 'rite', 'plea', 'country', 'Law', 'culture', 'saint', 'secondary', 'end', 'beautiful'], 'refulgent': ['summer'] }
```

Okay Now we have the dataset, let's start developing a function to generate sentences based on the next sentences. Do you see where I am going? Exactly it is easy just one word will be the key to the next word and until I reach a full stop (.) I will continue the iteration.

```python
    def generate_text(self, startword=None):
        '''
        Generate Sentences Based on Next Transition Word in the dataset
        :param startword: str
        :return:
        '''
        if not startword:
            startword = random.choice(list(self.dataset.keys()))
        sentence = startword.capitalize()
        print(startword)
        next_word = random.choice(self.dataset[startword])
        while next_word != '.':
            sentence += ' {}'.format(next_word)
            next_word = random.choice(self.dataset[next_word])
        sentence += '.'
        return sentence
```

I select a start word capitalize it in a sentence and keep on adding the next word from random values of that key until I find a period. and return the sentence. Ezy Pzy Right ? So to run the final answer let's create an object train it the data and generate some random sentences.

```python
if __name__ == '__main__':
    text = TextGenerator()
    text.generate_dataset()
    print(text.generate_text())
```

Some Example's of generated text with my dataset :

> 1.) Vitality , a community workforce … But it should also gives its constitution. \\
> 2.) Sizable percentage of the mistake is true lessons were dying , I would have to think of persuading young men. \\
> 3.) Wellesley spirit , you , resourceful , are all born into one of thousands of course that you four challenging for their buttons from a man ’ s sounding like her grasp of well-known people I made that would certainly not be bigger picture on that we can.
>
> ~ with love from My Weird Code

Final Code

```python
#!/usr/bin/env python
import glob
from collections import defaultdict
import nltk
import random



class TextGenerator:
    def __init__(self):
        self.dataset = defaultdict(list)

    def generate_dataset(self,file_pattern='datatask4/*.txt'):
        '''
        Generates Dataset from the nearby txt files
        :return: None
        '''

        for filename in glob.glob(file_pattern):
            with open(filename, 'r', encoding="utf-8") as file:
                file_data = file.readlines()
                file_words = nltk.word_tokenize(''.join(file_data))
                for i in range(len(file_words) -1):
                    self.dataset[file_words[i]].append(file_words[i + 1])
        self.dataset = dict(self.dataset)

    def generate_text(self, startword=None):
        '''
        Generate Sentences Based on Next Transition Word in the dataset
        :param startword: str
        :return:
        '''
        if not startword:
            startword = random.choice(list(self.dataset.keys()))
        sentence = startword.capitalize()
        print(startword)
        next_word = random.choice(self.dataset[startword])
        while next_word != '.':
            sentence += ' {}'.format(next_word)
            next_word = random.choice(self.dataset[next_word])
        sentence += '.'
        return sentence


if __name__ == '__main__':
    text = TextGenerator()
    text.generate_dataset()
    print(text.generate_text())
```

