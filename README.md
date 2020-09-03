# Machine Learning
[![ForTheBadge built-with-love](http://ForTheBadge.com/images/badges/built-with-love.svg)](https://github.com/NvsYashwanth)

![](https://badgen.net/badge/Code/Python/blue?icon=https://simpleicons.org/icons/python.svg&labelColor=cyan&label)    ![](https://badgen.net/badge/Library/ScikitLearn/blue?icon=https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg&labelColor=cyan&label)    ![](https://badgen.net/badge/Tools/pandas/blue?icon=https://simpleicons.org/icons/pandas.svg&labelColor=cyan&label)       ![](https://badgen.net/badge/Tools/numpy/blue?icon=https://upload.wikimedia.org/wikipedia/commons/1/1a/NumPy_logo.svg&labelColor=cyan&label)        ![](https://badgen.net/badge/Tools/matplotlib/blue?icon=https://upload.wikimedia.org/wikipedia/en/5/56/Matplotlib_logo.svg&labelColor=cyan&label)    ![](https://badgen.net/badge/icon/JupyterNotebook?icon=awesome&label)

`This repo hosts everything one needs to know about Machine Learning.
Every topic will be explained with code examples accompanied by Jupyter IPYNB notebooks on multiple datasets.
This is an on-going project.`

# Developers :two_men_holding_hands:
1. [NVS Yashwanth - Repository Maintainer & Owner](https://github.com/NvsYashwanth)
2. [Nikil Alakunta - Repository Contributor](https://github.com/Nikil99)

# CONTRIBUTION GUIDELINES :page_with_curl:
## How to contribute? :eyes:
1. Fork the repository
2. Create a new branch
3. Make the desired changes (add/delete/modify)
4. Make a pull request

## When to contribute? :eyes:
1. If there is any gramatical error in README files.
2. If you have a better explanation.
3. If there is any concept with wrong explanation or mistakes.
4. If any topic can be further explained with more emphasis on maths.
5. If you see a broken link, open an issue.

# If you find this repo useful, please star it and share :heart: :star:

# ***Contents*** :zap:
<ol>
 <li><a href='https://github.com/NvsYashwanth/Machine-Learning-Master#1-what-is-machine-learning'>What is Machine Learning?</a>
  <ul>
   <li><a href='https://github.com/NvsYashwanth/Machine-Learning-Master#arthur-samuel-1959'>Arthur Samuel</a></li>
   <li><a href='https://github.com/NvsYashwanth/Machine-Learning-Master#tom-mitchell1997'>Tom Mitchell</a></li>
  </ul>
 </li>
 
   <li><a href='https://github.com/NvsYashwanth/Machine-Learning-Master#2-types-of-machine-learning'>Types of Machine Learning</a>
   <ul>
    <li><a href='https://github.com/NvsYashwanth/Regression-Master#supervised-learning'>Supervised Learning</a>
     <ul>
       <li><a href='https://github.com/NvsYashwanth/Machine-Learning-Master/blob/master/README.md#classification-problems'>Classification Problems</a></li>
       <li><a href='https://github.com/NvsYashwanth/Machine-Learning-Master/blob/master/README.md#regression-problems'>Regression Problems</a></li>
     </ul>
    </li>
    <li><a href='https://github.com/NvsYashwanth/Regression-Master#unsupervised-learning'>Unsupervised Learning</a></li>
    <li><a href='https://github.com/NvsYashwanth/Regression-Master#reinforcement-learning'>Reinforcement Learning</a></li>
   </ul>
 </li>

 <li><a href='https://github.com/NvsYashwanth/Machine-Learning-Master#3-applications-of-machine-learning'>Applications of Machine Learning</a></li>
  
 <li><a href='https://github.com/NvsYashwanth/Machine-Learning-Master#4-machine-learning-life-cycle'>Machine Learning Life Cycle</a></li>
 
 <li><a href="https://github.com/NvsYashwanth/Machine-Learning-Master#5-bias-variance">Bias-Variance</a>
  <ul>
   <li><a href='https://github.com/NvsYashwanth/Machine-Learning-Master#bias-variance-trade-off'>Bias-Variance Tradeoff</a></li>
   <li><a href='https://github.com/NvsYashwanth/Machine-Learning-Master#model-fitting'>Model Fitting</a></li>
  </ul>
 </li>
 
  <li><a href="https://github.com/NvsYashwanth/Machine-Learning-Master#6-understanding-gradient-descent">Understanding Gradient Descent</a>
  <ul>
   <li><a href='https://github.com/NvsYashwanth/Machine-Learning-Master#cost-function'>Cost Function</a></li>
   <li><a href='https://github.com/NvsYashwanth/Machine-Learning-Master#gradient-descent'>Gradient Descent</a></li>
     <ul>
      <li><a href='https://github.com/NvsYashwanth/Machine-Learning-Master#learning-rate-%CE%B1'>Learning Rate α</a></li>
      <li><a href='https://github.com/NvsYashwanth/Machine-Learning-Master#normalization'>Normalization</a></li>
  </ul>
   
   <li><a href='https://github.com/NvsYashwanth/Machine-Learning-Master#the-optimization-procedure'>The optimization procedure</a></li>
  </ul>
 </li>
 
 <li><a href='https://github.com/NvsYashwanth/Machine-Learning-Master#7-types-of-gradient-descent'>Types of Gradient Descent</a>
   <ul>
   <li><a href='https://github.com/NvsYashwanth/Machine-Learning-Master#batch-gradient-descent'>Batch Gradient Descent</a></li>
   <li><a href='https://github.com/NvsYashwanth/Machine-Learning-Master#stochastic-gradient-descent-sgd'>Stochastic Gradient Descent (SGD)</a></li>
   <li><a href='https://github.com/NvsYashwanth/Machine-Learning-Master#mini-batch-gradient-descent'>Mini-batch Gradient Descent</a></li>
  </ul>
</li>
 
 
 <li><a href='https://github.com/NvsYashwanth/Machine-Learning-Master#8-beyond-the-first-order-optimization'>Beyond the first-order optimization</a></li>
 
</ol>

# More to come...  :heart:

# ***1. What is Machine Learning?***
* We shall look at the two most popular definitions of machine learning.

## Arthur Samuel (1959)
### ***`"Field of study which gives computers the ability to learn without beign explicitly programmed"`***

## Tom Mitchell(1997)
### ***`"A computer program is said to learn if its performance at a task T, as measured by a performance P, improves with experience E"`***

# ***2. Types of Machine Learning***
## Supervised Learning
* In Supervised Learning, input is provided as a labelled dataset. We build a model that can learn a function to perform input to output mapping.  ***There are 2 types of Supervised Learning problems***.

#### Classification Problems
* In this type, the model predicts a discrete value. The input data is usually a member of a particular class or group. For example, predicting whether the given image if of a dog or not. 

#### Regression Problems
* These problems are used for continuous data. For example, predicting the price of a piece of land in a city, given the area, location, number of rooms, etc.

## Unsupervised Learning
* This learning algorithm is completely opposite to Supervised Learning. In short, there is no complete and clean labelled dataset in unsupervised learning. Unsupervised learning is self-organized learning. Its main aim is to explore the underlying patterns and predicts the output.  Here we basically provide the machine with data and ask to look for hidden features and cluster the data in a way that makes sense.

## Reinforcement Learning
* It is neither based on supervised learning nor unsupervised learning. Moreover, here the algorithms learn to react to an environment on their own. It is rapidly growing and moreover producing a variety of learning algorithms. These algorithms are useful in the field of Robotics, Gaming etc.
* For a learning agent, there is always a start state and an end state. However, to reach the end state, there might be a different path. In Reinforcement Learning Problem an agent tries to manipulate the environment. The agent travels from one state to another. The agent gets the reward(appreciation) on success but will not receive any reward or appreciation on failure. In this way, the agent learns from the environment.

`NOTE: WE WILL NOT BE DISCUSSING REINFORCEMENT LEARNING`

# ***3. Applications of Machine Learning***
### `This part is self explanatory`
<p align='center'>
  <img src="https://github.com/NvsYashwanth/Machine-Learning-Master/blob/master/assets/ml%20applications.png">
</p>

# ***4. Machine Learning Life Cycle***
<p align='center'>
  <img src="https://github.com/NvsYashwanth/Machine-Learning-Master/blob/master/assets/ml%20life%20cycle.png">
</p>


# ***5. Bias-Variance***
#### Source: Medium Article by NVS Yashwanth (Original Author)

<p align='center'>
"Avoid the mistake of overfitting and underfitting."
</p>

***As a machine learning practitioner, it is important to have a good understanding of how to build effective models with high accuracy. A common pitfall in training a model is overfitting and underfitting.
Let us look at these topics so the next time you build a model, you would exactly know how to avoid the mistake of overfitting and underfitting.***

## Bias-Variance Trade-off
<p align='center'>
  "The two variables to measure the effectiveness of your model are bias and variance."
</p>

`Note: Please know that we are talking about the effectiveness of the model. If you are wondering about model validation, that is something we will discuss later.`

***Bias*** is the error or difference between points given and points plotted on the line in your training set.

***Variance*** is the error that occurs due to sensitivity to small changes in the training set.

<p align='center'>
  <img src="https://github.com/NvsYashwanth/Machine-Learning-Master/blob/master/assets/bias%20variance.jpg">
</p>

* I'll be explaining bias-variance further with the help of the image above. So please follow along. 

* To simply things, let us say, the error is calculated as the difference between predicted and observed/actual value. Now, say we have a model that is very accurate. This means that the error is very less, indicating a low bias and low variance. (As seen on the top-left circle in the image).

* If the variance increases, the data is spread more which results in lower accuracy. (As seen on the top-right circle in the image).

* If the bias increases, the error calculated increases. (As seen on the bottom-left circle in the image).

* High variance and high bias indicate that data is spread with a high error. (As seen on the bottom-right circle in the image).

* This is ***Bias-Variance Tradeoff***. Earlier, I defined bias as a measure of error between what the model captures and what the available data is showing, and variance being the error from sensitivity to small changes in the available data. A model having high variance captures random noise in the data.

* We want to find the best fit line that has low bias and low variance. (As seen on the top-left circle in the image).

### How do parameters affect our model?
* Model complexity keeps increasing as the number of parameters increase. This could result in overfitting, basically increasing variance and decreasing bias.

* Our aim is to come up with a point in our model where the decrease in bias is equal to an increase in variance. So how do we do this? Let us look at model fitting.



## Model Fitting
* We can find a line that represents the general direction of the points but might not represent every point in the dataset. This would be the best fit model.

### Why not use higher-order polynomials always?
* Good question. Sadly, the answer is no. By doing so, we would have created a model that fits our training data very well but fails to generalize beyond the training set (say any testing data that the model was not trained on). Therefore our model performs poorly on the test data giving rise to less accuracy. ***This problem is called over-fitting.*** We also say that the model has high variance and low bias.

* Similarly, we have another issue. ***It is called underfitting.*** It occurs when our model neither fits the training data nor generalizes on the new data (say any testing data that the model was not trained on). Our model is underfitting when we have high bias and low variance.

<p align='center'>
  <img src="https://github.com/NvsYashwanth/Machine-Learning-Master/blob/master/assets/Overfitted_Data.png">
</p>

***`The image above shows a blue line which is a polynomial regression line. The straight black line is that of a linear function. Although the polynomial function is a perfect fit, the linear function can be expected to generalize better. Thus, we can say that the polynomial function is overfitting, on the other hand, the straight line is the best fit. Think of an imaginary line that hardly passes through any of those points. That would be underfitting.`***

### How to overcome Underfitting & Overfitting for a regression model?
* ***To overcome underfitting*** or high bias, we can add new parameters to our model so that the model complexity increases thus reducing high bias.

* ***To overcome overfitting***, we could use methods like reducing model complexity and regularization.


# ***6. Understanding Gradient Descent***
<p align='center'>
"Let’s reach the global minimum."
</p>

***`Optimization algorithms are algorithms that are designed to converge to a solution. The solution here could be a local minimum or the global minimum by minimizing a cost function say ‘L’.`***

## Cost Function
* The cost function is a measure of how good our model is at making predictions. The shape of a cost function defines our goal of optimization.
* If the shape of the cost function is a convex function, our goal is to find the only minimum. This is relatively simpler cause there is no local minimum and we just need to converge to the global minimum.
* If the shape of the cost function is not a convex function, our goal is to find the lowest possible value in the neighborhood.

<p align='center'>
  <img src="https://github.com/NvsYashwanth/Machine-Learning-Master/blob/master/assets/covex.png">
</p>

## Gradient Descent
* ***`Gradient descent is a first-order iterative optimization algorithm used to minimize a function L, commonly used in machine learning and deep learning.`***

* It’s a ***first-order optimization algorithm*** because, in every iteration, the algorithm takes the first-order derivative for updating the parameters. Parameters refer to coefficients in a regression problem or the weights of a neural network. These parameters are updated by the gradient which gives the direction of the steepest ascent. In every iteration, this is performed by updating parameters in the opposite direction of the gradient computed for cost function L, w.r.t the parameters θ. The size of the update is determined by the step size called ***learning rate α***.

* To understand this algorithm, let us consider the case of a person trying to get to the bottom of the valley as quickly as possible.

* It is obvious that we take the direction of the steepest slope to move down quickly. Now, the question is how do we find this direction? Gradient Descent finds the same by measuring the local gradient of the error function and goes in the opposite direction of the gradient until we reach the global minimum.

<p align='center'>
  <img src="https://github.com/NvsYashwanth/Machine-Learning-Master/blob/master/assets/descent-1.png">
</p>

* As mentioned earlier, the algorithm calculates the gradient of the cost function w.r.t every parameter θ, which tells us the slope of our cost function at our current position (current parameter values) and the direction we should move to update our parameters. ***The size of our update is controlled by the learning rate.***

* The most commonly used learning rates are : 0.3, 0.1, 0.03, 0.01, 0.003, 0.001.

### Learning Rate α
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

### Normalization
* Before performing Gradient Descent, it is important to scale all our feature variables. Feature scaling plays a huge role in the shape of the cost function defined. The Gradient Descent algorithm converges quickly when normalization is performed or contours of cost function would be narrower and taller, which means it would take a longer time to converge.

* Normalizing data means scaling the data to attain (mean) ***μ=0*** with (standard deviation) ***σ=1***.

* Consider n feature variables. An instance xᵢ can be scaled as follows:

<p align='center'>
  <img src="https://github.com/NvsYashwanth/Machine-Learning-Master/blob/master/assets/feature%20scale.png">
</p>

* `One could use Scikit-Learn’s StandardScaler class to perform feature scaling.`

## The optimization procedure
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

# ***7. Types of Gradient Descent***
## Batch Gradient Descent
* Batch Gradient Descent uses a whole batch of training data at every training step. Thus it is very slow for larger datasets.

* The learning rate is fixed. In theory, if the cost function has a convex function, it is guaranteed to reach the global minimum, else the local minimum in case the loss function is not convex.

## Stochastic Gradient Descent (SGD)
* Batch Gradient descent uses a whole batch of the training set in every iteration of training. This is computationally expensive.

* Stochastic Gradient Descent (here stochastic means random), takes only a single instance (variance becomes large since we only use 1 example for every iteration of training) randomly and uses the same in every iteration of training. This is really fast but irregular patterns are observed due to the randomness. This randomness is however at times helpful if the cost function is irregular because it can jump out of the local minima with a better chance of finding the global minimum than in Batch Gradient Descent.

* We can use a learning rate scheduler set to an initial high value and then gradually decrease our learning rate over time so that we don't end up bouncing off from a minimum. If the learning rate is reduced too quickly, you may get stuck in a local minimum, or even end up halfway to the minimum. If the learning rate is reduced too slowly, you may jump around the minimum for a long time and end up with a suboptimal solution assuming you halt training too early.

* One can use scikit learns SGDRegressor class to implement this.

* An important thing to note is we must ensure our training data is shuffled at the beginning of each epoch so the parameters get pulled toward the global optimum, on average. If data is not shuffled, the SGD will not settle close to the global minimum, but instead will optimize label by label.

## Mini-batch Gradient Descent
* Mini-batch Gradient Descent computes the gradients on small random sets of instances called mini-batches. There is a better chance we can a bit closer to the minimum than Stochastic Gradient Descent but it may be harder for it to escape from local minima.

* The batch size can be a power of 2 like 32,64, etc.

* Shuffling data like in the case of SGD is recommended to avoid a pre-existing order.

* Mini-batch Gradient Descent is faster than Batch Gradient Descent since less number of training examples are used for every training step. It also generalizes better.

* The drawback is, it is difficult to converge as one might jump around the minimum region due to the noise present. These oscillations are the reason we need a learning rate decay to decrease the learning rate as move closer to the minimum.

# ***8. Beyond the first-order optimization***
* As mentioned earlier, Gradient Descent is a first-order optimization algorithm meaning it only measures the slope of the cost function curve but not the curvature. ( Curvature means the degree by which a curve or a surface deviates from being a straight line or a plane respectively).

* So how nature of the function or its curvature measured? It is determined by the second-order derivative. The curvature affects our training.

* ***If the Second-order derivative is equal to 0***, the curvature is said to be linear.

* ***If the Second-order derivative is greater than 0***, the curvature is said to be moving upward.

* ***If the Second-order derivative is less than 0***, the curvature is said to be moving downwards.

# Resources :heart:
## Books
<ol>
 <li><a href="https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/">Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd Edition</a></li>
</ol>

## Topic wise links
<ol>
 <li><a>Basic Terminology of Machine Learning</a>
  <ul>
   <li><a href="https://www.javatpoint.com/machine-learning">What is Machine Learning? What are the types?</a></li>
   <li><a href="https://www.javatpoint.com/applications-of-machine-learning">Applications Of Machine Learning</a></li>
   <li><a href="https://www.javatpoint.com/machine-learning-life-cycle">Machine learning Life cycle</a></li>
  </ul>
 </li>
 
  <li><a>Bias-variance</a>
  <ul>
   <li><a href="https://medium.com/analytics-vidhya/bias-variance-tradeoff-b4c6c181030d">Bias-Variance Tradeoff - Analytics Vidhya</a></li>
  </ul>
 </li>
</ol>

  <li><a>Gradient Descent</a>
  <ul>
   <li><a href="https://medium.com/analytics-vidhya/gradient-descent-and-beyond-ef5cbcc4d83e">Understanding Gradient Descent - Analytics Vidhya</a></li>
  </ul>
 </li>
</ol>

## Websites
<ol>
 <li><a href="https://machinelearningmastery.com/">Machine Learning Mastery</a></li>
</ol>

## Newsletters

## Datasets

## Youtube Channels

