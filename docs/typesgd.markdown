---
layout: page
title: 6.1. Types Of Gradient Descent
permalink: /types-of-gradient-descent/
parent: 6. Understanding Gradient Descent
nav_order: 1
---


# ***6.1. Types of Gradient Descent***
#### Source: Medium Article by NVS Yashwanth (Original Author)

## ***Batch Gradient Descent***
* Batch Gradient Descent uses a whole batch of training data at every training step. Thus it is very slow for larger datasets.

* The learning rate is fixed. In theory, if the cost function has a convex function, it is guaranteed to reach the global minimum, else the local minimum in case the loss function is not convex.

## ***Stochastic Gradient Descent (SGD)***
* Batch Gradient descent uses a whole batch of the training set in every iteration of training. This is computationally expensive.

* Stochastic Gradient Descent (here stochastic means random), takes only a single instance (variance becomes large since we only use 1 example for every iteration of training) randomly and uses the same in every iteration of training. This is really fast but irregular patterns are observed due to the randomness. This randomness is however at times helpful if the cost function is irregular because it can jump out of the local minima with a better chance of finding the global minimum than in Batch Gradient Descent.

* We can use a learning rate scheduler set to an initial high value and then gradually decrease our learning rate over time so that we don't end up bouncing off from a minimum. If the learning rate is reduced too quickly, you may get stuck in a local minimum, or even end up halfway to the minimum. If the learning rate is reduced too slowly, you may jump around the minimum for a long time and end up with a suboptimal solution assuming you halt training too early.

* One can use scikit learns SGDRegressor class to implement this.

* An important thing to note is we must ensure our training data is shuffled at the beginning of each epoch so the parameters get pulled toward the global optimum, on average. If data is not shuffled, the SGD will not settle close to the global minimum, but instead will optimize label by label.

## ***Mini-batch Gradient Descent***
* Mini-batch Gradient Descent computes the gradients on small random sets of instances called mini-batches. There is a better chance we can a bit closer to the minimum than Stochastic Gradient Descent but it may be harder for it to escape from local minima.

* The batch size can be a power of 2 like 32,64, etc.

* Shuffling data like in the case of SGD is recommended to avoid a pre-existing order.

* Mini-batch Gradient Descent is faster than Batch Gradient Descent since less number of training examples are used for every training step. It also generalizes better.

* The drawback is, it is difficult to converge as one might jump around the minimum region due to the noise present. These oscillations are the reason we need a learning rate decay to decrease the learning rate as move closer to the minimum.