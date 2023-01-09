---
layout: post
title: Deriving categorical cross entropy and softmax
categories: [machine-learning, math]
tags: [machine-learning, theory, math, calculus, derivatives, backpropagation, neural-networks, information-theory]
image:
  path: thumbnail.png
  alt: Categorical cross entropy and softmax
math: true
---

## Introduction

Recently, on the Pytorch discussion someone asked the question about the derivation of categorical cross entropy and softmax. To my surprise, I couldn't find a good answer to it. So I thought it would be a good idea to write a blog post about it. So here it is.

### Cross-Entropy from information theory

To understand the categorical cross entropy from a bit of information theory background. The definition of cross entropy in information theory:
> The cross entropy is the average number of bits required to encode an event drawn from a probability distribution if the other probability distribution is used to define the code.
{: .prompt-info }
A layman definition for me will be
> How much surprise I will get about the outcome of an event if I think that the custom distribution of the event models the outcome and is equal to the actual distribution.
{: .prompt-info }

#### Cross-Entropy loss

We would want to minimise this loss/suprise/average number of bits required.

The cross entropy loss is equal to the negative log likelihood of the actual distribution. So if we have a distribution $ p $ and we want to model it with a distribution $ q $ then the cross entropy loss is equal to

$$ \mathcal{L}(s, y) = - \sum_{i=1}^{C} y_i \log s_i$$

where $ s $ is the predicted distribution i.e logits after the softmax function and $ y $ is the actual distribution in the form of one hot encoded vector.
