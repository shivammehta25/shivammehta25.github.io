---
layout: post
title: Keras Callbacks
date: 2019-09-07 16:45 +0100
categories: [machine-learning]
tags: [machine-learning, deep-learning, neural-networks]
image:
  path: thumbnail.jpeg
  alt: Keras Callbacks
---

## (Migrated from old blog)

Alright, I was on Summer Vacations to my home, India. Therefore, I was inactive for a month or so but now let's get back and begin again.

So что такое ( What is ) Callbacks, as the term refers callbacks is a way to drive the training of your neural network to a defined function at a specific interval ( thats how I define it)

As per Keras official documentation

> A callback is a set of functions to be applied at given stages of the training procedure. You can use callbacks to get a view on internal states and statistics of the model during training. You can pass a list of callbacks (as the keyword argument `callbacks`) to the `.fit()` method of the `Sequential` or `Model` classes. The relevant methods of the callbacks will then be called at each stage of the training.
>
> [https://keras.io/callbacks/](https://keras.io/callbacks/)

Which is more or less the same as my definition. So how to apply it.

```python
tf.keras.callbacks.Callback
```

We will be using this class of callback and using one of Object Oriented Principles i.e Inheritance we will be creating our custom one that will have let's say we will stop training when the accuracy reaches 85 %.

```python
class CallBack(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('acc') > 0.85:
            print('Stopping Training as Acurracy Is over 85%')
            self.model.stop_training = True

callback = CallBack()
```

Let's take our all-time favorite some MNIST database. Umm lets take fashion one

```python
from keras.datasets import fashion_mnist
((trainX, trainY), (testX, testY)) = fashion_mnist.load_data()
trainX = trainX.astype('int32')
trainY = trainY.astype('int32')
testX = testX.astype('int32')
testX = testX.astype('int32')
trainX = trainX/255.0
testX = testX/255.0
```

And lets make a basic model to tell what this thing is

```python
model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28,28)),
        keras.layers.Dense(200, activation=tf.nn.relu),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

The final trick is to add the callback parameter in the fit method of the model object

```python
model.fit(trainX, trainY, verbose=1, epochs=5, callbacks=[callback])
```

Now the training will stop after 2-3 epochs approx as the model will gain over 85% accuracy as we mentioned in the on\_epoch\_end method

What it did basically is that after each end of an epoch is took the model to the object of callback and ran the method with the parameters and when it saw the accuracy greater than 85% it ran the model.stop\_training=True

It was a basic introduction to callbacks and we will go through the documentation of it to learn more and post about it later.

Hope you understood something like I did.
