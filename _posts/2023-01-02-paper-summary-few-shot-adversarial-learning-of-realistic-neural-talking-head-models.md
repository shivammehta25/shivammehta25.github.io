---
layout: post
title: Paper summary - Few shot adversarial learning of realistic neural talking head models
date: 2019-11-08 15:23 +0100
categories: [deep-learning, paper-summary]
tags: [machine-learning, deep-learning, neural-networks, generative-models, probabilistic-machine-learning, python, GAN]
image:
  path: thumbnail.jpeg
  alt: Paper summary - Few shot adversarial learning of realistic neural talking head models
math: true
---

## (Migrated from old blog)

Recently, I saw new research which I was amazed by the results and idea behind it. I spent some time trying to read it and hence I am sharing my notes and understanding of it. The research orignally can be found at [https://arxiv.org/abs/1905.08233](https://arxiv.org/abs/1905.08233) The research team was from SkolTech : [Egor Zakharov](https://arxiv.org/search/cs?searchtype=author&query=Zakharov%2C+E), [Aliaksandra Shysheya](https://arxiv.org/search/cs?searchtype=author&query=Shysheya%2C+A), [Egor Burkov](https://arxiv.org/search/cs?searchtype=author&query=Burkov%2C+E), [Victor Lempitsky](https://arxiv.org/search/cs?searchtype=author&query=Lempitsky%2C+V)

## **Introduction**

- Recent work have shown that we can generate several realistic human head images can be shown by training generative convolutional neural networks over large dataset of images, but in practical scenarios this has to be learned from a few image view of a person.
- This paper introduces a system with few shot capabilities, it implements meta learning on large dataset of videos and is then able to frame few and one shot learning of head models on previously unseen people as adversarial training problems.

### **Problem with Synthesising realistic talking head sequences**

- Human heads have high photometric and geometric and kinematic complexity, the problems like modelling mouth cavity, hair and garments.
- To overcome such challenges there are algorithms but creating head sequences with amount of motion, head rotation etc is still hard to achieve.
- To do such things with ConvNets we have to train them on large networks where both generator and discriminators have tens of millions of parameters and thus, they require several minute long videos or large dataset of Photographs and GPU to train.

## **Working:**

- The few shot learning ability is obtained through extensive pre-training on a large corpus of talking head videos corresponding to different speakers with diverse appearance. 
- The system simulates few-shot learning tasks and learns to transform landmark positions into realistically- looking personalised photographs.
- After that few photographs of new person sets up a new adversarial learning problem with high-capacity generator and discriminator pre-trained via meta learning.
- These new adversarial problem converges to the state that generates realistic and personalized images after a few training step.

## **META LEARNING STAGE :**

### **Architecture:**

- xi is the i’th video sequence and xi(t) is the t’th frame and yi(t) is the landmark image computed for xi(t).
- The embedder $ \mathcal{E}(\xi(s), y_i(s); \psi) $  takes a video from an associated landmark image and maps these inputs into an N-Dimensional vector $eˆi$, that contains video specific information that is invariant to pose and mimics in a particular frame.

![Working of Embedder during Meta Learning](Few_shot_learning_paper_1.png)
_Working of Embedder during Meta Learning_

- The Generator G(yi(t), eˆi ; ψ, P) takes the landmark image for the video frame not seen by the embedder, the video embeddings from embedder and outputs a synthesised video frame. The generator is trained to maximise the similarity between its outputs and ground truth frames. The parameters of generator are split into two frames one person generic (ψ) and the other person specific (ψˆi). During meta learning the person specific (ψˆi) are predicted from the embedding vector eˆi using a trainable projection matrix P: ψˆ i = P eˆi

![Working of Generator in the Network with Emedding Parameters Initialized](Few_shot_learning_paper_2.png)
_Working of Generator in the Network with Emedding Parameters Initialized_

- The Discriminator r D(xi(t), yi(t), i; θ,W, w0, b), takes a video-frame xi(t) an associated landmark image yi(t) and the index of training sequence i. (θ,W, w0, b) are parameters of discriminator. The discriminator predicts a single scalar realism score (r) that indicates whether the input frame is a real frame of the ith video sequence and matches the input pose yi(t).

![Working of Discriminator in the Network](Few_shot_learning_paper_3.png)
_Working of Discriminator in the Network_

Now our model has meta information and time to fine tune it for specific video or photo using few shot learning

## Few-Shot Learning

### **Embedding Part:**

- The parameters of all three networks are trained in an adversarial fashion by simulating episodes of K-shot learning (K=8 in paper).
- In each episode they randomly draw training video sequence and a single frame from that sequence, similar to that frame more K frames were drawn randomly from the same sequence. 
- Then they computed the estimate embeddings eˆi of the ith video embeddings by averaging the eˆi for every k frame.

![Generation of Embeddings during Few Shot Learning Phase](Few_shot_learning_paper_4.png)
_Generation of Embeddings during Few Shot Learning Phase_

### **Generation:**

xˆi(t) is regenerated by the generator based on these estimated embeddings. 

### **Discriminator:**

The discriminator first maps its input to N dimensional vector and then computes the realism score, with this vector and W matrix which contains the embeddings thats corresponds to the individual video. (Thus there are two video embeddings one computed by embedder and other corresponds to the column of the matrix W in discriminator)

After that we train the Generator and Discriminator on the few available images, using the same adversarial objective as in the meta learning stage. 

### Adversarial Objective and Training 

$$ \begin{aligned} \mathcal{L}\left(\phi, \psi, \mathbf{P}, \theta, \mathbf{W}, \mathbf{w}{0}, b\right) &=\mathcal{L}_{\mathrm{CNT}}(\phi, \psi, \mathbf{P})+\ \mathcal{L}_{\mathrm{ADV}}\left(\phi, \psi, \mathbf{P}, \theta, \mathbf{W}, \mathbf{w}{0}, b\right)+\mathcal{L}_{\mathrm{MCH}}(\phi, \mathbf{W}) \end{aligned} $$

Where,

 $ \mathcal{L}_{\mathrm{CNT}} $ constant term loss measures distance between ground truth image and reconstructed image.

$ \mathcal{L}_{\mathrm{ADV}} $ corresponds to the realism score computed by the discriminator, which needs to be maximised

$ \mathcal{L}_{\mathrm{MCH}} $ encourages the similarity between two types of embeddings by penalising the L1 difference between eˆi and Wi.

$$ \begin{aligned} \mathcal{L}{\mathrm{DSC}}\left(\phi, \psi, \mathbf{P}, \theta, \mathbf{W}, \mathbf{w}{0}, b\right) &=\ \max \left(0,1+D\left(\hat{\mathbf{x}}{i}(t), \mathbf{y}{i}(t), i ; \phi, \psi, \theta, \mathbf{W}, \mathbf{w}{0}, b\right)\right)+\\ \max \left(0,1-D\left(\mathbf{x}{i}(t), \mathbf{y}{i}(t), i ; \theta, \mathbf{W}, \mathbf{w}{0}, b\right)\right) \end{aligned} $$

L(DSC) compares the realism of fake and real example and then updates the discriminator parameters.

The training proceeds by alternating updates of the embedder, the generator that minimise the loss L(CNT) ,L(ADV), L(MCH) and with the update of discriminator that minimise the loss L(DSC).

## **Evaluation Metrics:**

They fine tuned all models on few shot learning set of size T for a person which was not seen durning meta-learning stage. After few shot learning the evaluation is performed on hold out part of the same sequence.
For evaluation they uniformly sampled 50 videos from VoxCeleb test sets and 32 hold out frames for each of these videos.

FID: Frechet inception distance measuring perceptual realism
SSIM (structured similarity): measuring low-level similarity to the ground truth images
CSIM (Cosine Similarity): between embedding vectors of the state of the art face recognition network for measuring identity mismatch ( VGGFace)

Also performed a user study where three different video sequences were shown to user and out of which one was fake produced by model. And user was asked to distinguish. And USER accuracy was measured. ( thus lower the better for the model).

![Evaluation Matrix of the Proposed Model](Few_shot_learning_paper_5.png)
_Evaluation Matrix of the Proposed Model_

## **Conclusion, Results and Examples**

They presented a framework for meta-learning of Adversarial Generative Model, which was able to train highly realistic virtual talking heads in form of deep generator networks.

![Results in Comparison to other Models](Few_shot_learning_paper_6.png)
_Results in Comparison to other Models_

![Few Shot Learning Results](Few_shot_learning_paper_7.png)
_Few Shot Learning Results_

![Example of Living Portraits](Few_shot_learning_paper_8.png)
_Example of Living Portraits_

### **More Examples at their Youtube Video**

{% include embed/youtube.html id='p1b5aiTrGzY' %}
_Youtube Video Explaining More about the working and examples_

## **Side Note:**

This technology is amazing but It can have adverse or bad effects since fake news and bad information will be more convincing with such technologies but on the brighter side we should use such technologies for empowering the humans like it can be used to regenerate memories of someone who is dead. It's our jobs to make world a better place lets use technology the right way.
