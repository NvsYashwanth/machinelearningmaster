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
</ol>

## Websites
<ol>
 <li><a href="https://machinelearningmastery.com/">Machine Learning Mastery</a></li>
</ol>

## Newsletters

## Datasets

## Youtube Channels

