---
layout: post
title: Deriving categorical cross entropy and softmax
categories:
- machine-learning
- math
tags:
- machine-learning
- theory
- math
- calculus
- derivatives
- backpropagation
- neural-networks
- information-theory
image:
  path: thumbnail.jpeg
  alt: Categorical cross entropy and softmax
math: true
date: 2023-01-10 09:51 +0100
---
## Introduction

Recently, on the Pytorch discussion forum, someone asked the question about the derivation of categorical cross entropy and softmax. So I thought it would be a good idea to write a blog post about it with more details to it.

### Cross-Entropy from information theory

The idea of this section is to understand the categorical cross entropy from a bit of information theory background. The definition of cross-entropy:
> The cross entropy is the average number of bits required to encode an event drawn from a probability distribution if the other probability distribution is used to define the code.
{: .prompt-info }
A layman definition relating to entropy for me will be
> How much surprise on average I will get about the outcome of an event if I think that the custom distribution of the event models the true distribution.
{: .prompt-info }

## Cross-Entropy loss

We would want to minimize this loss/surprise/average number of bits required. The cross-entropy loss is equal to the negative log-likelihood of the actual distribution. So if we have a distribution $ p $ and we want to model it with a distribution $ q $ then the cross entropy loss is equal to

$$ \mathcal{L}(s, y) = - \sum_{i=1}^{C} y_i \log s_i$$

where $ s $ is the predicted distribution i.e logits after the softmax function and $ y $ is the actual distribution in the form of one hot encoded vector.

## Softmax

Softmax is a generalization of the logistic function to multiple dimensions. It is a function that takes as input a vector of K real numbers and normalizes it into a probability distribution consisting of K probabilities. The output of the softmax function is a vector of probabilities that sum to 1, thus making it a probability distribution as the outputs are non-negative and sum to 1.

Mathematically softmax is a vector function that maps an input vector to an output vector. $ s : \mathbb{R}^n \to \mathbb{R}^n $

$$ \text{softmax}(x) = \frac{e^{x / T}}{\sum_{j=1}^{C} e^{x_j / T}} $$

where $T = 1$ generally. This is the temperature of the softmax which I will talk about in more detail in a different blog post later, for now, let's assume that $T = 1$ and the function becomes

$$ s_i = \frac{e^{x_i}}{\sum_{j=1}^{C} e^{x_j}} \tag{1}$$

Why we talked about softmax because we need the softmax and its derivative to get the derivative of the cross-entropy loss.

### Derivation of softmax

When we talk about the derivative of a vector function we talk about its jacobian. The jacobian of softmax is a matrix of all first-order partial derivatives of the softmax function.

$$ \nabla \text{softmax} = \left(\begin{array}{cccc}
\frac{\partial s_1}{\partial x_1} & \frac{\partial s_1}{\partial x_2} & \cdots & \frac{\partial s_1}{\partial x_n} \\
\frac{\partial s_2}{\partial x_1} & \frac{\partial s_2}{\partial x_2} & \cdots & \frac{\partial s_2}{\partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial s_n}{\partial x_1} & \frac{\partial s_n}{\partial x_2} & \cdots & \frac{\partial s_n}{\partial x_n} \tag{2}
\end{array}\right)$$

Since the output of each element depends on the sum of all other elements the output of the non-diagonal elements will not be zero.

Let's begin by taking the log of softmax, we often use $\log$ and $\ln$ interchangeably. Generally in information theory, we stick to $\log$ with base 2 but in deep learning math, we more frequently use $\log$ with base $e$ i.e natural log $ = \ln$.
$$ \begin{align*}
\log s_i &= \log \left( \frac{e^{x_i}}{\sum_{j=1}^{C} e^{x_i}} \right) \\
 &= \log \left( e^{x_i} \right) - \log \left( \sum_{j=1}^{C} e^{x_i} \right)  \tag{log(a/b) = log a - log b} \\
&= x_i - \log \left( \sum_{j=1}^{C} e^{x_i} \right)
 \end{align*} $$
Now differentiating the above equation with respect to some arbitrary $x_k$ we get

$$ \begin{align*}
\frac{\partial \log s_i}{\partial x_k} &= \frac{\partial x_i}{\partial x_k} - \frac{\partial \log \left( \sum_{j=1}^{C} e^{x_i} \right)}{\partial x_k} \tag{3} \\
\end{align*} $$

The first element of RHS
$$
 \frac{\partial x_i}{\partial x_k} =  \begin{cases}
1 & x_i = x_k \\
0 & \text{otherwise}
\end{cases}  = \mathbb{1}(x_i = x_k)  = \delta_{ik} $$
where $\mathbb{1}(x_i = x_k)$ is the indicator function which in this case can also represented as $\delta_{ik}$ which is the [Kronecker delta function](https://en.wikipedia.org/wiki/Kronecker_delta).

And the second element of RHS can be further simplified as

$$
\begin{align*}
\frac{\partial \log \left( \sum_{j=1}^{C} e^{x_i} \right)}{\partial x_k} &= \frac{\partial \log \left( \sum_{j=1}^{C} e^{x_j} \right)}{\partial x_k} \\
&= \frac{1}{\sum_{j=1}^{C} e^{x_j}} \frac{\partial \left( \sum_{j=1}^{C} e^{x_j} \right)}{\partial x_k} \\
&= \frac{1}{\sum_{j=1}^{C} e^{x_j}} \sum_{j=1}^{C} \frac{\partial e^{x_j}}{\partial x_k} \\
&= \frac{1}{\sum_{j=1}^{C} e^{x_j}} \sum_{j=1}^{C} e^{x_j} \frac{\partial x_j}{\partial x_k} \\
&= \frac{1}{\sum_{j=1}^{C} e^{x_j}} \sum_{j=1}^{C} e^{x_j} \mathbb{1}(x_j = x_k) \\
&= \frac{1}{\sum_{j=1}^{C} e^{x_j}} \sum_{j=1}^{C} e^{x_j} \delta_{jk} \\
&= \frac{e^{x_k}}{\sum_{j=1}^{C} e^{x_j}} \\
&= s_k
\end{align*}
$$

So the final equation becomes:

$$
\begin{align*}
\frac{\partial \log s_i}{ \partial x_k} &= \delta_{ik}  - s_k  \\
\frac{1}{s_i} \frac{\partial s_i}{\partial x_k} &= \delta_{ik}  - s_k  \\
\frac{\partial s_i}{\partial x_k} &= s_i \left(\delta_{ik}  - s_k \right) \tag{4} \\
\end{align*}
$$

We can now use the $\frac{\partial s_i}{\partial x_k}$ to get the jacobian of softmax in equation (2).

$$ \nabla \text{softmax} = \left(\begin{array}{cccc}
 s_1\left(1 - s_1\right) & -s_1s_2 & \cdots & -s_1s_n \\
-s_2s_1 & s_2(1-s_2) & \cdots & -s_2s_n \\
\vdots & \vdots & \ddots & \vdots \\
-s_ns_1 & -s_ns_2 & \cdots & s_n(1-s_n)
\end{array}\right) \tag{5}$$

Eq. 5 is the backward/reverse/jacobian of the softmax function.

## Differentiating cross-entropy loss

We differentiate cross entropy with respect to the input to the softmax x_k as the local derivative of softmax defined in Eq. 4 will help us to get a trivial solution for cross entropy's local derivative.

$$
\begin{align*}
 \mathcal{L}(s, y) &= - \sum_{i=1}^{C} y_i \log s_i \\
 \frac{\partial \mathcal{L}(s, y)}{\partial x_k} &= - \sum_{i=1}^{C} y_i \frac{\partial \log s_i}{\partial x_k} \\
  &= - \sum_{i=1}^{C} \frac{y_i}{s_i} \frac{\partial s_i}{\partial x_k} \\
  &= - \sum_{i=1}^{C} \frac{y_i}{s_i} s_i \left(\delta_{ik}  - s_k \right) \tag{from 4}\\
  &= - \sum_{i=1}^{C} y_i \left(\delta_{ik}  - s_k \right) \\
  &= - \sum_{i=1}^{C} y_i \delta_{ik} + y_i s_k
\end{align*}
$$

when $i = k$ only then the first term will become = $y_k$

$$
\begin{align*}
\frac{\partial \mathcal{L}(s, y)}{\partial x_k} &= - y_k + \sum_{i=1}^{C} y_k s_k
\end{align*}
$$

where $\sum_{i=1}^C y_k = 1$ as $y_k$ is a one-hot vector.

$$
\begin{align*}
\frac{\partial \mathcal{L}(s, y)}{\partial x_k} &= - y_k + s_k \\
\frac{\partial \mathcal{L}(s, y)}{\partial x_k} &= s_k - y_k
\end{align*}
$$

In the vectorized (numpy array/torch tensor) form it can simply be written as

$$
\begin{align*}
\frac{\partial \mathcal{L}(s, y)}{\partial x} &= s - y \tag{6}
\end{align*}
$$

which is the derivation of the categorical cross-entropy loss function.

I hope this helps with the theoretical understanding of the softmax and cross-entropy loss function. I have tried to keep it as simple as possible and explain every single step. If you have any questions or suggestions please let me know. I will try to answer them as soon as possible.
