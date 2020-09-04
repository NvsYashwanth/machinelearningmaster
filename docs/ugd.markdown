---
layout: page
title: Understanding Gradient Descent
permalink: /ugd/
nav_order: 6
---
# ***6. Understanding Gradient Descent***
#### Source: Medium Article by NVS Yashwanth (Original Author)

<p align='center'>
"Let’s reach the global minimum."
</p>

***`Optimization algorithms are algorithms that are designed to converge to a solution. The solution here could be a local minimum or the global minimum by minimizing a cost function say ‘L’.`***

## ***Cost Function***
* The cost function is a measure of how good our model is at making predictions. The shape of a cost function defines our goal of optimization.
* If the shape of the cost function is a convex function, our goal is to find the only minimum. This is relatively simpler cause there is no local minimum and we just need to converge to the global minimum.
* If the shape of the cost function is not a convex function, our goal is to find the lowest possible value in the neighborhood.

<p align='center'>
  <img src="https://github.com/NvsYashwanth/Machine-Learning-Master/blob/master/assets/covex.png">
</p>

## ***Gradient Descent***
* ***`Gradient descent is a first-order iterative optimization algorithm used to minimize a function L, commonly used in machine learning and deep learning.`***

* It’s a ***first-order optimization algorithm*** because, in every iteration, the algorithm takes the first-order derivative for updating the parameters. Parameters refer to coefficients in a regression problem or the weights of a neural network. These parameters are updated by the gradient which gives the direction of the steepest ascent. In every iteration, this is performed by updating parameters in the opposite direction of the gradient computed for cost function L, w.r.t the parameters θ. The size of the update is determined by the step size called ***learning rate α***.

* To understand this algorithm, let us consider the case of a person trying to get to the bottom of the valley as quickly as possible.

* It is obvious that we take the direction of the steepest slope to move down quickly. Now, the question is how do we find this direction? Gradient Descent finds the same by measuring the local gradient of the error function and goes in the opposite direction of the gradient until we reach the global minimum.

<p align='center'>
  <img src="https://github.com/NvsYashwanth/Machine-Learning-Master/blob/master/assets/descent-1.png">
</p>

* As mentioned earlier, the algorithm calculates the gradient of the cost function w.r.t every parameter θ, which tells us the slope of our cost function at our current position (current parameter values) and the direction we should move to update our parameters. ***The size of our update is controlled by the learning rate.***

* The most commonly used learning rates are : 0.3, 0.1, 0.03, 0.01, 0.003, 0.001.

### ***Learning Rate α***
* ***A high learning rate*** results in large step sizes. Though there is a chance to reach the bottom-most point quickly, we risk overshooting the global minimum as the slope of the hill is constantly changing.

<p align='center'>
  <img src="https://github.com/NvsYashwanth/Machine-Learning-Master/blob/master/assets/lr%20large.png">
</p>

* ***A low learning rate*** results in small step sizes. Thus, we move in the opposite direction of the gradient precisely. The drawback here is the time required for calculating the gradient. So it will take us a very long time to converge (to reach the bottom-most point).

<p align='center'>
  <img src="https://github.com/NvsYashwanth/Machine-Learning-Master/blob/master/assets/lr%20small.png">
</p>

* As mentioned before, our goal is to reach a global minimum. But at times when our cost function has an irregular curve (mostly in the case of deep learning neural networks), with random initialization in place, one might reach a ***local minimum***, which is not as good as the global minimum. One way to overcome this is by using the concept of momentum.

<p align='center'>
  <img src="https://github.com/NvsYashwanth/Machine-Learning-Master/blob/master/assets/pitfalls%20descent.png">
</p>

* `The learning rate affects how quickly our model can converge. Thus the right value means lesser time for us to train the model. This is crucial because lesser training time means lesser GPU run-time.`

### ***Normalization***
* Before performing Gradient Descent, it is important to scale all our feature variables. Feature scaling plays a huge role in the shape of the cost function defined. The Gradient Descent algorithm converges quickly when normalization is performed or contours of cost function would be narrower and taller, which means it would take a longer time to converge.

* Normalizing data means scaling the data to attain (mean) ***μ=0*** with (standard deviation) ***σ=1***.

* Consider n feature variables. An instance xᵢ can be scaled as follows:

<p align='center'>
  <img src="https://github.com/NvsYashwanth/Machine-Learning-Master/blob/master/assets/feature%20scale.png">
</p>

* `One could use Scikit-Learn’s StandardScaler class to perform feature scaling.`

## ***The optimization procedure***
* Before getting into further details let us define partial derivatives.
* ***`In Gradient Descent we compute the gradient of the cost function w.r.t every parameter. This means that we calculate how much the cost function varies when either of the parameters changes. This is called a partial derivative.`***

* Consider a cost function L, with parameters θ. It can be denoted as L(θ). Our goal is to minimize this cost function by finding the best parameter θ values.

1. We initialize our parameters θ with random values.

2. We choose a learning rate α and perform feature scaling (normalization).

3. Upon every iteration of the algorithm, we calculate the gradient of cost function w.r.t every parameter and update them as follows:

<p align='center'>
  <img src="https://github.com/NvsYashwanth/Machine-Learning-Master/blob/master/assets/gradient.png">
</p>

<p align='center'>
  <img src="https://github.com/NvsYashwanth/Machine-Learning-Master/blob/master/assets/update%20step.png">
</p>

* `The negative sign in the optimization step shows that we update our parameters in the opposite direction of the gradient computed for cost function L, w.r.t the parameters θ.`

* ***If the gradient is less than 0***, we increase the parameters by the value of the gradient multiplied by learning rate α.

* ***If the gradient is greater than 0***, we decrease the parameters by the value of the gradient multiplied by learning rate α.

* The above steps are repeated until the cost function converges. Now, by the convergence we mean, the gradient of the cost function would be equal to 0.

<p align='center'>
  <img src="https://github.com/NvsYashwanth/Machine-Learning-Master/blob/master/assets/convergence.png">
</p>