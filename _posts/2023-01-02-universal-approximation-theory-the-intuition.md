---
layout: post
title: Universal approximation theory - The intuition
date: 2021-06-10 17:02 +0100
categories: [deep-learning, math]
tags: [deep-learning, theory, math, calculus, derivatives, backpropagation, neural-networks]
image:
  path: thumbnail.jpeg
  alt: Universal approximation theory - The intuition
math: true
---

## (Migrated from old blog)

Recently, I attended a course on Deep Learning and found this very nice intuition for how the Universal Approximation Theorem works.

# What is Universal Approximation Theorem?

**Universal Approximation Theorem** states that a neural network with a single hidden layer and finite number of neurons can approximate any continuous functions in $ \mathbb{R} ^n$ (under mild conditions).

## Quick revision of a neuron in a neural network

![Neuron in Neural Networks](neuron.jpeg)
_Neuron in Deep Learning_

It takes a vector of input $ x_i $ multiplies it with weights $ W $ adds a bias $ b $ and applies a non linearity $ g\left( \cdot\right) $ and gives an output $ a_i $.

Basically,

$$ \begin{align*}
z_i &= W_i \cdot x_i + b \\
a_i &= g(z_i)
\end{align*} $$

where $ g(x) $ is the activation function. For our use we will be using a **Re**ctified **L**inear **U**nit or **ReLU** activation function which is defined as

$$ z = \mathbf{ReLU}(x) = \max(0, x)$$

 and its derivative is


$$
f(x)=
\begin{cases}
0 & \text{if } x < 0 \\
1 & \text{if } x >= 0 \\
\end{cases}
$$

**ReLU** has lot of benefits but we will not be discussing about it rather we will try to approximate some functions with its help.

Lets plot a simple neuron with ReLU activation we can say that

$$ a = \max(w.x + b, 0) $$

where w = 1 and b= 0 will look like

Images from desmos play around with them [here](https://www.desmos.com/calculator/5whjspwgcm)

![Relu Activation](Relu-Activation.png)
_Rectified Linear Unit activation function or ReLU_

where w = 5 and b = -1 we will get the graph as

![Neuron with W=5 and b=-1](Neuron_with_W_5_and_b_-1.png)
_Neuron with W=5 and b=-1_

So, we can make lot of straight with different slopes and biases lines with just a single neuron with ReLU activation.

## Objective Function

Now let's try to create a complex function that will require more than one neuron, a function that looks like this

![Objective function with multiple neurons](Objective-function-multiple-neurons.png)
_Objective function with multiple neurons_

So now, we cannot model with objective function with a single neuron rather we will stack two neurons vertically basically something like this

![Find out parameters to this Neural Network](Objective-function.jpeg)
_Find out parameters to this Neural Network_

where $ a_i $ is the activations of the $ \text{neuron}_i $

Spend some time on desmos link before looking at the answer you will figure it out yourself:
[https://www.desmos.com/calculator/nfx8gtciyy](https://www.desmos.com/calculator/nfx8gtciyy)

## Solution

$$ a_1 = ReLU\left(w_1 \cdot x + b_1\right) $$

where $ w_1 = 1 \ \text{and} \ b_1 = 0 $

![w = 1 and b = 0](w_1_b_0.png)

w = 1 and b = 0

and

$$ a_2 = ReLU\left(w_2 \cdot x + b_2\right) $$

 where $ w_2 = 1 \ \text{and} \ b_2 = -1 $

![w=1 and b=-1](w_1_b_-1.png)
_w=1 and b=-1_

Now we take the difference of these two activations i.e

$$ f(x) = a_1 - a_2 $$

 and we model our objective function look at this plot and in desmos to play around this approximation.
[https://www.desmos.com/calculator/fiblrs3lnm](https://www.desmos.com/calculator/fiblrs3lnm)

![f(x) = a1 - a2](a_1-a_2.png)
_f(x) = a1 - a2_

Similarly, by combining many networks in parallel like this, we can model any piecewise constant function and this family is rich enough to approximate any continuous function.

Thus, showing us how the **Universal Approximation Theorem** works intuitively.

I learned this intuition from Lennart Svensson and Jakob Lindqvist, Department of Electrical Engineering, Chalmers University of Technology, Sweden
