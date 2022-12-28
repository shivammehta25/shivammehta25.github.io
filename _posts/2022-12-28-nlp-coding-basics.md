---
layout: post
title: NLP coding basics (before LLMs)
date: 2019-04-10 23:40 +0100
categories: [nlp]
tags: [machine-learning, nlp]
image:
  path: thumbnail.jpeg
  alt: Machine Learning Theory
---
## (Migrated from old blog)

This is from the video tutorials of sentdex :
{% include embed/youtube.html id='FLZvOKSCkxY' %}

I am just like you trying to learn things so I will be sharing my code here with you all. For a better explanation, see the video tutorials of sentdex, that guy is awesome.

So importing everything

```python
import nltk
nltk.download() # and downloading all
```

## Tokenizing the Sentences

Splitting sentences and words from the body of text into different tokens

```python
example_sentence = 'Hello Shivam Mehta! How are you? How is it going? everything is alright lets learn something new? I am fine thank you. It was pleasure to meet you. I am from India. Currently studying in ITMO University in Saint Petersburg, Russia.'
print('sentence: {}'.format(sent_tokenize(example_sentence)))
print('words : {}'.format(word_tokenize(example_sentence)))
```

```bash
# Output
sentence: ['Hello Shivam Mehta!', 'How are you?', 'How is it going?', 'everything is alright lets learn something new?', 'I am fine thank you.', 'It was pleasure to meet you.', 'I am from India.', 'Currently studying in ITMO University in Saint Petersburg, Russia.']
words : ['Hello', 'Shivam', 'Mehta', '!', 'How', 'are', 'you', '?', 'How', 'is', 'it', 'going', '?', 'everything', 'is', 'alright', 'lets', 'learn', 'something', 'new', '?', 'I', 'am', 'fine', 'thank', 'you', '.', 'It', 'was', 'pleasure', 'to', 'meet', 'you', '.', 'I', 'am', 'from', 'India', '.', 'Currently', 'studying', 'in', 'ITMO', 'University', 'in', 'Saint', 'Petersburg', ',', 'Russia', '.']
```

## Stopwords

For example, you may wish to completely cease analysis if you detect words that are commonly used sarcastically, and stop immediately. Sarcastic words, or phrases are going to vary by lexicon and corpus. For now, we'll be considering stop words as words that just contain no meaning, and we want to remove them. You can do this easily, by storing a list of words that you consider to be stop words.

```python
from nltk.corpus import stopwords
print(stopwords.words('english'))
```

```bash
# Output

['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
```

Filtering our example with stopwords

```python
english_stop_words = set(stopwords.words('english'))
filtered_words = filter(lambda x: x not in english_stop_words, word_tokenize(example_sentence.lower()))
print(list(filtered_words))
```

```bash
#Output
['hello', 'shivam', 'mehta', '!', '?', 'going', '?', 'everything', 'alright', 'lets', 'learn', 'something', 'new', '?', 'fine', 'thank', '.', 'pleasure', 'meet', '.', 'india', '.', 'currently', 'studying', 'itmo', 'university', 'saint', 'petersburg', ',', 'russia', '.']
```

We see lot of useless words are trimmed only some meaningful one's are remaining that has some essence in the corpora.

## Stemming

Stemming is the way of normalizing the data. That is making playing, plays, played and all other endings to just play. Sentdex example :

I was taking a ride in the car.
I was riding in the car.

This sentence means the same thing. in the car is the same. I was is the same. the ing denotes a clear past-tense in both cases, so is it truly necessary to differentiate between ride and riding, in the case of just trying to figure out the meaning of what this past-tense activity was?

```python
from nltk.stem import PorterStemmer
ps = PorterStemmer()
stemming_words = ["python","pythoner","pythoning","pythoned","pythonly" ]
print(list(map(ps.stem, stemming_words )))
```

```bash
# Output
['python', 'python', 'python', 'python', 'pythonli']
```

Stemming our Example Text we get

```python
print(list(map(ps.stem,filtered_words)))
```

```bash
# Output
['hello', 'shivam', 'mehta', '!', '?', 'go', '?', 'everyth', 'alright', 'let', 'learn', 'someth', 'new', '?', 'fine', 'thank', '.', 'pleasur', 'meet', '.', 'india', '.', 'current', 'studi', 'itmo', 'univers', 'saint', 'petersburg', ',', 'russia', '.']
```

## Part of Speech Tagging ( POS Tagging)

This means labeling words in a sentence as nouns, adjectives, verbs...etc. Even more impressive, it also labels by tense, and more. Here's a list of the tags, what they mean, and some examples:

```bash
POS tag list:

CC coordinating conjunction
CD cardinal digit
DT determiner
EX existential there (like: "there is" ... think of it like "there exists")
FW foreign word
IN preposition/subordinating conjunction
JJ adjective 'big'
JJR adjective, comparative 'bigger'
JJS adjective, superlative 'biggest'
LS list marker 1)
MD modal could, will
NN noun, singular 'desk'
NNS noun plural 'desks'
NNP proper noun, singular 'Harrison'
NNPS proper noun, plural 'Americans'
PDT predeterminer 'all the kids'
POS possessive ending parent's
PRP personal pronoun I, he, she
PRP$ possessive pronoun my, his, hers
RB adverb very, silently,
RBR adverb, comparative better
RBS adverb, superlative best
RP particle give up
TO to go 'to' the store.
UH interjection errrrrrrrm
VB verb, base form take
VBD verb, past tense took
VBG verb, gerund/present participle taking
VBN verb, past participle taken
VBP verb, sing. present, non-3d take
VBZ verb, 3rd person sing. present takes
WDT wh-determiner which
WP wh-pronoun who, what
WP$ possessive wh-pronoun whose
WRB wh-abverb where, when
```

Code To Mark It , we will try to implement PunktSentenceTokenizer, which is a trainable tokenizer. It is pretrained and we can train it more so I will use nltk.corpus and get one random corpus just to train it more I can not do it and it will work almost the same in this case. But still for the sake of learning why not ?

```python
from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import state_union
train_text = state_union.raw('2005-GWBush.txt')
sentence_tokenizer = PunktSentenceTokenizer(train_text)
tokenized_sentence = sentence_tokenizer.tokenize(example_sentence)
print(tokenized_sentence)
pos_tagged = list(map(nltk.pos_tag, [word_tokenize(sentence) for sentence in tokenized_sentence])) = list(map(nltk.pos_tag, [word_tokenize(sentence) for sentence in tokenized_sentence]))
print(pos_tagged)
# Clever trick to get it all in one line Otherwise you can also write like
# for i in tokenized_sentence:
#            words = nltk.word_tokenize(i)
#            tagged = nltk.pos_tag(words)
#            print(tagged)
```

```bash
# Output
[[('Hello', 'NNP'), ('Shivam', 'NNP'), ('Mehta', 'NNP'), ('!', '.')], [('How', 'WRB'), ('are', 'VBP'), ('you', 'PRP'), ('?', '.')], [('How', 'WRB'), ('is', 'VBZ'), ('it', 'PRP'), ('going', 'VBG'), ('?', '.')], [('everything', 'NN'), ('is', 'VBZ'), ('alright', 'JJ'), ('lets', 'NNS'), ('learn', 'VBP'), ('something', 'NN'), ('new', 'JJ'), ('?', '.')], [('I', 'PRP'), ('am', 'VBP'), ('fine', 'JJ'), ('thank', 'NN'), ('you', 'PRP'), ('.', '.')], [('It', 'PRP'), ('was', 'VBD'), ('pleasure', 'NN'), ('to', 'TO'), ('meet', 'VB'), ('you', 'PRP'), ('.', '.')], [('I', 'PRP'), ('am', 'VBP'), ('from', 'IN'), ('India', 'NNP'), ('.', '.')], [('Currently', 'RB'), ('studying', 'VBG'), ('in', 'IN'), ('ITMO', 'NNP'), ('University', 'NNP'), ('in', 'IN'), ('Saint', 'NNP'), ('Petersburg', 'NNP'), (',', ','), ('Russia', 'NNP'), ('.', '.')]]
```

## Chunking

It means to group words into hopefully meaningful chunks. One of the main goals of chunking is to group into what are known as "noun phrases." These are phrases of one or more words that contain a noun, maybe some descriptive words, maybe a verb, and maybe something like an adverb. The idea is to group nouns with the words that are in relation to them.

```python
chunk_gram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""
chunk_parser = nltk.RegexpParser(chunk_gram)
for item in pos_tagged:
    chunked = chunk_parser.parse(item)
    chunked.draw()
    print(chunked)
```

```bash
# Output
(S (Chunk Hello/NNP Shivam/NNP Mehta/NNP) !/.)
(S How/WRB are/VBP you/PRP ?/.)
(S How/WRB is/VBZ it/PRP going/VBG ?/.)
(S
  everything/NN
  is/VBZ
  alright/JJ
  lets/NNS
  learn/VBP
  something/NN
  new/JJ
  ?/.)
(S I/PRP am/VBP fine/JJ thank/NN you/PRP ./.)
(S It/PRP was/VBD pleasure/NN to/TO meet/VB you/PRP ./.)
(S I/PRP am/VBP from/IN (Chunk India/NNP) ./.)
(S
  Currently/RB
  studying/VBG
  in/IN
  (Chunk ITMO/NNP University/NNP)
  in/IN
  (Chunk Saint/NNP Petersburg/NNP)
  ,/,
  (Chunk Russia/NNP)
  ./.)
```

## Chinking

Even after chunking if you want to remove some words, we can use chinking. (funny name)

```python
chink_gram = r"""Chunk: {<.*>+}
                        }<VB.? |IN|DT|TO>+{"""
chink_parser = nltk.RegexpParser(chink_gram)
for item in pos_tagged:
    chinked = chink_parser.parse(item)
    print(chinked)
#     chinked.draw()
```

```bash
#Output
(S (Chunk Hello/NNP Shivam/NNP Mehta/NNP !/.))
(S (Chunk How/WRB) are/VBP (Chunk you/PRP ?/.))
(S (Chunk How/WRB) is/VBZ (Chunk it/PRP) going/VBG (Chunk ?/.))
(S
  (Chunk everything/NN)
  is/VBZ
  (Chunk alright/JJ lets/NNS)
  learn/VBP
  (Chunk something/NN new/JJ ?/.))
(S (Chunk I/PRP) am/VBP (Chunk fine/JJ thank/NN you/PRP ./.))
(S
  (Chunk It/PRP)
  was/VBD
  (Chunk pleasure/NN)
  to/TO
  meet/VB
  (Chunk you/PRP ./.))
(S (Chunk I/PRP) am/VBP from/IN (Chunk India/NNP ./.))
(S
  (Chunk Currently/RB)
  studying/VBG
  in/IN
  (Chunk ITMO/NNP University/NNP)
  in/IN
  (Chunk Saint/NNP Petersburg/NNP ,/, Russia/NNP ./.))
```

## Named Entity Recognition

Recognition weather the word in the sentence is an orgranization or person or a geo location. NLTK provides a good solution not the best but not bad either.

```bash
NE Type and Examples
ORGANIZATION - Georgia-Pacific Corp., WHO
PERSON - Eddy Bonte, President Obama
LOCATION - Murray River, Mount Everest
DATE - June, 2008-06-29
TIME - two fifty a m, 1:30 p.m.
MONEY - 175 million Canadian Dollars, GBP 10.40
PERCENT - twenty pct, 18.75 %
FACILITY - Washington Monument, Stonehenge
GPE - South East Asia, Midlothian
```

```python
for sentence in pos_tagged:
    entity_recog = nltk.ne_chunk(sentence) # we can use another parameters binary=True that will just say weather it is named entity or not
    print(entity_recog)
```

```bash
# Output
(S (PERSON Hello/NNP) (PERSON Shivam/NNP Mehta/NNP) !/.)
(S How/WRB are/VBP you/PRP ?/.)
(S How/WRB is/VBZ it/PRP going/VBG ?/.)
(S
  everything/NN
  is/VBZ
  alright/JJ
  lets/NNS
  learn/VBP
  something/NN
  new/JJ
  ?/.)
(S I/PRP am/VBP fine/JJ thank/NN you/PRP ./.)
(S It/PRP was/VBD pleasure/NN to/TO meet/VB you/PRP ./.)
(S I/PRP am/VBP from/IN (GPE India/NNP) ./.)
(S
  Currently/RB
  studying/VBG
  in/IN
  (ORGANIZATION ITMO/NNP University/NNP)
  in/IN
  (GPE Saint/NNP Petersburg/NNP)
  ,/,
  (GPE Russia/NNP)
  ./.)
```

## Lemmatize

It is like stemming but sometimes more useful as it can yield meaningful words.

```python
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
for sentence in pos_tagged:
    lemmatized_words = [lemmatizer.lemmatize(word[0]) for word in sentence]
    print(lemmatized_words)
```

```bash
# Output
['Hello', 'Shivam', 'Mehta', '!']
['How', 'are', 'you', '?']
['How', 'is', 'it', 'going', '?']
['everything', 'is', 'alright', 'let', 'learn', 'something', 'new', '?']
['I', 'am', 'fine', 'thank', 'you', '.']
['It', 'wa', 'pleasure', 'to', 'meet', 'you', '.']
['I', 'am', 'from', 'India', '.']
['Currently', 'studying', 'in', 'ITMO', 'University', 'in', 'Saint', 'Petersburg', ',', 'Russia', '.']
```

Some examples from sentdex that we should consider:

```python
print(lemmatizer.lemmatize("better", pos="a"))
print(lemmatizer.lemmatize("best", pos="a"))
print(lemmatizer.lemmatize("run"))
print(lemmatizer.lemmatize("run",'v'))
```

```bash
# Output
good
best
run
run
```

## Wordnet Corpus

One of the most useful in nltk for synonyms antonyms, definitions, examples, etc.

```python
from nltk.corpus import wordnet
synset = wordnet.synsets('program')
print(synset)
```

```bash
# Output
[Synset('plan.n.01'), Synset('program.n.02'), Synset('broadcast.n.02'), Synset('platform.n.02'), Synset('program.n.05'), Synset('course_of_study.n.01'), Synset('program.n.07'), Synset('program.n.08'), Synset('program.v.01'), Synset('program.v.02')]
```

```python
print(synset[0].name())
print(synset[0].lemmas())
print(synset[0].definition())
print(synset[0].examples())
```

```bash
# Output
good.n.01
[Lemma('good.n.01.good')]
benefit
['for your own good', "what's the good of worrying?"]
```

For Synonyms and Antonyms:

```python
synset = wordnet.synsets('good')
synonym = []
antonym = []
for syn in synset:
    for l in syn.lemmas():
        synonym.append(l.name())
        if l.antonyms():
            antonym.extend([x.name() for x in l.antonyms()])
print(set(synonym))
print(set(antonym))
```

```bash
# Output
{'right', 'in_force', 'salutary', 'ripe', 'honorable', 'sound', 'serious', 'goodness', 'trade_good', 'full', 'soundly', 'expert', 'safe', 'in_effect', 'dear', 'dependable', 'good', 'adept', 'practiced', 'beneficial', 'respectable', 'near', 'honest', 'secure', 'upright', 'skilful', 'thoroughly', 'unspoilt', 'commodity', 'estimable', 'well', 'just', 'unspoiled', 'undecomposed', 'effective', 'proficient', 'skillful'}
{'evil', 'badness', 'evilness', 'ill', 'bad'}
```

Finding Similarities between words

```python
w1 = wordnet.synset('bike.n.01')
w2 = wordnet.synset('bicycle.n.01')
print(w1.wup_similarity(w2))
w1 = wordnet.synset('bike.n.01')
w2 = wordnet.synset('car.n.01')
print(w1.wup_similarity(w2))
w1 = wordnet.synset('bike.n.01')
w2 = wordnet.synset('cat.n.01')
print(w1.wup_similarity(w2))
w1 = wordnet.synset('bike.n.01')
w2 = wordnet.synset('boat.n.01')
print(w1.wup_similarity(w2))
```

```bash
# Output
0.7272727272727273
0.9166666666666666
0.32
0.6956521739130435
```
