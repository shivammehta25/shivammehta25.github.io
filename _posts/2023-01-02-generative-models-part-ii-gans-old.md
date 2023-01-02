---
layout: post
title: Generative Models - Part II GANs (old)
date: 2019-10-21 15:00 +0100
categories: [deep-learning]
tags: [machine-learning, deep-learning, neural-networks, generative-models, probabilistic-machine-learning, python, GAN]
image:
  path: thumbnail.jpeg
  alt: Generative models part I Variational autoencoders
math: true
---

## New Edit (Jan 2023)

This post does not talks about other properties of GANs like backward KL divergence, mode collapse, etc. I will write a new post on that soon.

## (Migrated from old blog)

Cool, So far we learned that PixelCNN generates a probabilistic density tractable function and we optimize the likelihood of the generated data.
Also, now we know that Variational AutoEncoders (VAE) defines an intractable density function with latent **z**. Which we cannot optimize directly, hence we optimize the lower bound of that likelihood.

But in the case of GAN, aka Generative Adversarial Networks, we give up on explicitly modeling density rather focus on ability to sample the data. So, instead of working with explicit density function, GAN's choose game-theory approach, learn from the training distribution through 2 player game.

When we want to sample from a very high dimensional and complex distribution, we sample from a simpler distribution ( Gaussian, or random) and then learn transformations to the training distribution.

And what do we do when we want to model a complex network ? **NEURAL NETWORK** for the win!!!

So, we have a 2 player network :

- **Generated Network:** Tries to fool the discriminator by generating real looking images
- **Discriminator Network:** Tried to distinguish between the ground truth and the generated image.

![Working of Generative Adversarial Networks](thumbnail.jpeg)

Working of Generative Adversarial Networks

MinMax Objective Function:
$$
\min_{\theta{g}} \max_{\theta{d}}\left[\mathbb{E}_{x \sim p_{data }} \log D_{\theta_{d}}(x)+\mathbb{E}_{z \sim p(z)} \log \left(1-D_{\theta_{d}}\left(G_{\theta_{g}}(z)\right)\right)\right]
$$
Where $ {\log D_{\theta_{d}}(x)} $ is the discriminator's output for the real data where $x $ is sampled from $ p_{data} $ , and $ D_{\theta_{d}}\left(G_{\theta_{g}}(z)\right) $ is the discriminator's output for Generated Images sampled from $ p(z) $ and $ \theta_g \text{ and } \theta_d $ are parameters of generator and discriminator respectively.

## Training

So, what we do is we alternated between

- **Gradient Ascent of a Discriminator network:**
    $$ \max_{\theta_d} \left[ \mathbb{E}_{x \sim p_{data}} \log D_{\theta_d} (x) + \mathbb{E}_{z \sim p(z)} \log \left(1-D_{\theta_d}\left(G_{\theta_g}(z)\right)\right) \right] $$
- **Gradient Descent on the Generator Network:**
    $$ \min_{\theta_g} \mathbb{E}_{z \sim p(z)} \log \left( 1 - D_{\theta_d} \left( G (z) \right) \right) $$

But in practice the generator optimization doesn't works well because of the way its slopes work, look at the image:

![Why GAN does not optimises well](GAN_2_2.png)
_Why GAN does not optimises well_

So, if we look at loss landscape $ \log ( 1 - D(G(x)) $ we see that our generators work only when we already have a very bad sample, but when the sample are bad it is very easy for discriminator and thus the gradient is flat, So it makes it hard to learn, So the idea is to find a different objective function.

So instead, we apply Gradient Ascent i.e instead of minimising likelihood of discriminator being correct, now we maximize the likelihood of discriminator being wrong and surprisingly it works much better in practice, so the function becomes: $$ \max_{\theta_g} \mathbb{E}_{z \sim p(z)} \log D_{\theta_d}\left( G_{\theta_g} \left( z) \right) \right) $$

![How optmization function changes by Stanford](GAN_2_3.png)
_How optmization function changes by Stanford_

So the final Algorithm looks like this:

![GAN Algorithm](GAN_2_4.png)
_GAN Algorithm_

Once you have the generator trained you can use the generator to generate real life images.

The results of GAN were very realistic and GAN is widely used from research to industries.

For a Convolutional GAN, general approaches include,
The generator is an upsampling network with fractionally-strided convolutions and Discriminator is a convolutional network.

**A lot of GAN training Tips and Tricks that works are: [https://github.com/soumith/ganhacks](https://github.com/soumith/ganhacks)**.
**There are lot of different type of GAN check this link :
**[**https://github.com/hindupuravinash/the-gan-zoo**](https://github.com/hindupuravinash/the-gan-zoo)
