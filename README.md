# Regression Methods

# `CONTRIBUTION GUIDELINES` :page_with_curl:
## How to contribute? :eyes:
1. Fork the repository
2. Make the desired changes (add/delete/modify)
3. Make a pull request

## When to contribute? :eyes:
1. If there is any gramatical error in README files.
2. If you have a better explanation.
3. If there is any concept with wrong explanation or mistakes.
4. If any topic can be expanded further with more emphasis on maths.

# ***Contents*** :zap:
<ol>
 
 <li><a href='https://github.com/NvsYashwanth/Regression-Master/blob/master/README.md#1-linear-regression'>Linear Regression</a>
  <ul>
   <li><a href='https://github.com/NvsYashwanth/Regression-Master#prediction-model'>Prediciton Model</a></li>
   <li><a href='https://github.com/NvsYashwanth/Regression-Master#training'>Training</a></li>
   <li><a href='https://github.com/NvsYashwanth/Regression-Master#cost-function'>Cost Function</a></li>
  </ul>
 </li>
 
  <li><a href='https://github.com/NvsYashwanth/Regression-Master#2-types-of-gradient-descnet'>Types Of Gradient Descent</a>
  <ul>
   <li><a href='https://github.com/NvsYashwanth/Regression-Master#batch-gradient-descent'>Batch Gradient Descent</a></li>
   <li><a href='https://github.com/NvsYashwanth/Regression-Master#stochastic-gradient-descent'>Stochastic Gradient Descent</a></li>
   <li><a href='https://github.com/NvsYashwanth/Regression-Master#mini-batch-gradient-descnet'>Mini-Batch Gradient Descent</a></li>
  </ul>
 </li>
 
 <li><a href='https://github.com/NvsYashwanth/Regression-Master#3-polynomial-regression'>Polynomial Regression</a>
 </li>
 
 <li><a href='https://github.com/NvsYashwanth/Regression-Master#4-learning-curves'>Learning curves</a>
 </li>
 
 <li><a href='https://github.com/NvsYashwanth/Regression-Master#5-regularization-models'>Regularization Models</a>
 </li>
</ol>


# ***1. Linear Regression***
### `Linear Regression is a supervised machine learning algorithm that predicts a continuous output`

## Prediction Model
* We define the linear relationship for a given dependent variable `y` as ouput and the independent variable `x`as input. The model tries to fit a straight line by predicting the coefficients `m`(slope) and `b`(intercept or the bias term) where variables `m` and `b` are optimized to produce accurate predictions.
<p align='center'>
  <img src="https://github.com/NvsYashwanth/Sales-prediction/blob/master/assets/simple%20regression.png">
</p>

***The predictions can be modelled as*** :
<p align='center'>
  <img src='https://github.com/NvsYashwanth/Sales-prediction/blob/master/assets/general%20pred.png'>
</p>

***In vectorized form*** :
<p align='center'>
  <img src='https://github.com/NvsYashwanth/Sales-prediction/blob/master/assets/general%20vector%20pred.png'>
</p>

## Training
Let us consider the follow set of data points :
```
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
X = 2 * np.random.rand(100, 1)
y = -1 + 2 * X + np.random.randn(100, 1)
plt.subplot(111)
plt.scatter(X,y)
plt.title('Data Points')
plt.legend(['Random Data']) 
plt.show()
```
The output of the above code : 
<p align='center'>
  <img src='https://github.com/NvsYashwanth/Regression-Master/blob/master/assets/data%20points%20linear%20reg%20example.png'>
</p>


### ***For small number of features***
### Normal Equation
* Also regarded as the closed form.
<p align='center'>
  <img src='https://github.com/NvsYashwanth/Sales-prediction/blob/master/assets/normal%20eq.png'>
</p>

`Complexity is : O(n**3)`

```
X_b=np.c_[np.ones((100, 1)),X]
thetas=np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
X_new=np.array([[0],[2]])
X_new_b=np.c_[np.ones((2, 1)),X_new]
y_pred_normaleq=X_new_b.dot(thetas)
plt.subplot(111)
plt.scatter(X,y,color='orange')
plt.plot(X_new,y_pred_normaleq,'r-')
plt.title('Normal Equation Fit')
plt.legend(['Best Fit Line'])
print(f"Normal Equation ---> Intercept : {thetas[0]} and Coefficient : {thetas[1]}\n\n")
plt.show()
```
The output of the above code : 
<p align='center'>
  <img src='https://github.com/NvsYashwanth/Regression-Master/blob/master/assets/normal%20eq%20fit.png'>
</p>


### Pseudoinverse
* LinearRegression() of the scikit learn computes pseudoinverse using `SVD (Singular Value Decomposition)`. This is computationally efficient than Normal Equation especially when the matrix X is not invertible (singular).

`Complexity is : O(n**2)`

```
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X,y)
y_pred_skl=regressor.predict(X)
plt.subplot(111)
plt.scatter(X,y)
plt.plot(X,y_pred_skl,'r-')
plt.title('Scikit Learn : Linear Regression')
plt.legend(['Best Fit Line'])
print(f"Scikit Learn Linear Regression ---> Intercept : {regressor.intercept_} and Coefficient : {regressor.coef_}\n\n")
plt.show()
```
The output of the above code : 
<p align='center'>
  <img src='https://github.com/NvsYashwanth/Regression-Master/blob/master/assets/sklearn%20linear%20reg.png'>
</p>

***`As you can see the result is same for both Normal Equation and Scikit-Learn SVD method.`**

### ***For large number of features ~100,000***
### Gradient Descent
* Gradient descent is an optimization algorithm used to minimize some function (here in our case cost function that describes error) iteratively by moving in the negative direction of the gradient. 
* To undestand this algorithm, let us consider the case of a man trying to get to the bottom of the valley. 
<p align='center'>
  <img src='https://github.com/NvsYashwanth/Sales-prediction/blob/master/assets/mountain.png'>
</p>

* The best way to get down quickly is to move in the direction of the steepest slope. Gradient Desent achieves the same by measuring the local gradient of the error function and goes in the direction of descending gradient. Our goal is to reach the global minimum.
<p align='center'>
  <img src='https://github.com/NvsYashwanth/Sales-prediction/blob/master/assets/descent-1.png'>
</p>

* The algorithm calculates the gradient or partial derivatives of the cost function w.r.t every parameter. This new gradient tells us the slope of our cost function at our current position (current parameter values) and the direction we should move to update our parameters. The size of our update is controlled by the learning rate.
* With a high learning rate which means larger steps, we risk overshooting the global minimum since the slope of the hill is constantly changing. 
<p align='center'>
  <img src='https://github.com/NvsYashwanth/Sales-prediction/blob/master/assets/lr%20large.png'>
</p>

* With a very low learning rate, we can move in the direction of the negative gradient precisely, but calculating the gradient is time-consuming, so it will take us a very long time to converge (to reach at the bottom most point).
<p align='center'>
  <img src='https://github.com/NvsYashwanth/Sales-prediction/blob/master/assets/lr%20small.png'>
</p>

* As mentioned before our goal is to reach the global minimum. But at times when our cost function has irregular cruve, with random initialization in place, one might reach a local minimum, which is not as good as the global minimum. 
<p align='center'>
  <img src='https://github.com/NvsYashwanth/Regression-Master/blob/master/assets/pitfalls%20descent.png'>
</p>

* One way to overcome this is by using the concept of [momentum](https://distill.pub/2017/momentum/).
## Cost Function
* Cost function tells us how good our model is at making predictions. It has its own curve and gradient.
* For the Linear Regression model, we define the cost funciton MSE (Mean Square Error) that measures the average squared differnce between actual and predicted values. Our goal is to minimize this cost function in order to improve the accuracy of the model. It is a convex function, which means that line joining any given 2 points on this never has a local minimum. Only a global minimum.

<p align='center'>
  <img src='https://github.com/NvsYashwanth/Regression-Master/blob/master/assets/MSE%20equation.png'>
</p>

* It is important to note that feature scaling plays a huge role in the shape of the cost function defined. As you can see, on the left the Gradient Descent algorithm goes straight toward the minimum quickly, whereas on the right it first goes in a direction almost orthogonal to the direction of the global minimum. It will eventually reach the minimum, but it will take a long time to converge.

<p align='center'>
  <img src='https://github.com/NvsYashwanth/Sales-prediction/blob/master/assets/mse%20fig.png'>
</p>

* When using Gradient Descent, one should ensure that all features have a similar scale (e.g., using Scikit-Learn’s StandardScaler class), or else it will take much longer to converge.


# ***2. Types Of Gradient Descnet***
* Before getting into further details let us define ***partial derivatives***. 

`In Gradient Descnet we compute the gradient of the cost function w.r.t every parameter. This means that we calculate how much the cost function varies when either of the parameter change. This is called partial derivatives.`


## Batch Gradient Descent
* Batch Gradient Descent uses whole batch of training data at every step of training step. Thus it is very slow for larger datasets.
<p align='center'>
  <img src='https://github.com/NvsYashwanth/Regression-Master/blob/master/assets/partial%20derivates.png'>
</p>

The above given method of computation is slow. Thus we can consider the vectorized form as follows :
<p align='center'>
  <img src='https://github.com/NvsYashwanth/Regression-Master/blob/master/assets/gradient%20vector.png'>
</p>

Upon finding the gradient, we can subtract the same to go in the negative direction and eventually reach downhill. The amount by which the same changes is defined by the learning rate ( Refer the section on gradient descent intro to understand learning rate )
<p align='center'>
  <img src='https://github.com/NvsYashwanth/Regression-Master/blob/master/assets/learning%20step.png'>
</p>

In order to understand the same let us look at the following example :
```
import seaborn as sns
import matplotlib.pyplot as plt
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Defining a list of tuples of learning rates and subplots
lr = [(0.05,131),(0.5,132),(1,133)]

# number  of iterations
epochs = 100

# Training examples
m = 100

# Parameters 
# Considering simple regression
theta = np.random.randn(2,1) # random initialization

# 50 data points
X_b = np.c_[np.ones((100, 1)), X] 

# Taking 2 points (extremes) to form our regression line 
X_new = np.array([[0], [2]])

# Adding a constant to each since X(0) is 1.
X_new_b = np.c_[np.ones((2, 1)), X_new] # add x0 = 1 to each instance


# The main loop
for lr in lr:
    for iteration in range(epochs):
        gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
        theta = theta - lr[0] * gradients
        y_predict = X_new_b.dot(theta)
        plt.figure(1,figsize=(15,5))
        plt.subplot(lr[1])
        plt.plot(X_new, y_predict, "r")
        plt.plot(X, y, "b.")
        plt.ylim((0,14))
        plt.title(f"Learning Rate : {lr[0]}")

```
The output of the above code : 
<p align='center'>
  <img src='https://github.com/NvsYashwanth/Regression-Master/blob/master/assets/code%20example%20lrs.png'>
</p>

In order to find an appropriate learning rate, one can use something like [grid search](https://scikit-learn.org/stable/modules/grid_search.html).

## Stochastic Gradient Descent
* Batch Gradient descent uses whole batch of training set in every iteration of training. This is computationally expensive.
* Stochastic Gradient Descent (here stochastic means random), takes only a single instance randomly and uses the same to in every iteration to train. This is much faster, but irregular patterns are observed due to the randomness.
* This randomness is however at times helpful if the cost function is irregular. Because it can jump out of the local minimas with a better chance of finding global minimum than in Batch Gradient Descent.
* We can use a learning rate scheduler set to a inital high value and then gradually decrease our learning rate over time so that we dont end up bouncing off from minimum. If the learning rate is reduced too quickly, you may get stuck in a local minimum, or even end up frozen halfway to the minimum. If the learning rate is reduced too slowly, you may jump around the minimum for a long time and end up with a suboptimal solution if you halt training too early.
* One can use scikit learns SGDRegressor class to implement this.
* An important thing to note is we must ensure our training data is shuffled at the begining of each epoch so the parameters get pulled toward the global optimum, on average. If data is not shuffled, the SGD will not settle close to the global minimum, but instead will optimize label by label.

## Mini-batch Gradient Descnet
* Mini-batch Gradient Descnet computes the gradients on small random sets of instances called mini-batches. There is a better chance we can a bit closer to the minimum than Stochastic Gradient Descent but it may be harder for it to escape from local minima.

***Comparision between Batch, Stochastic and Mini-Batch***
* Though Batch GD is computationally slow its path actually stops at the minimum, while both Stochastic GD and Mini-batch GD bounce but would also reach the minimum if you used a good learning schedule.
<p align='center'>
  <img src='https://github.com/NvsYashwanth/Regression-Master/blob/master/assets/grad%20algos.png'>
</p>

# ***3. Polynomial Regression***
* Polynomial Regression is a regression algorithm that models the relationship between a dependent(y) and independent variable(x) as nth degree polynomial. The Polynomial Regression equation is given below:
<p align='center'>
  <img src='https://github.com/NvsYashwanth/Regression-Master/blob/master/assets/poly%20reg.png'>
</p>

* If non-linearity is present in our data, then we can opt Polynomial Regression.
<p align='center'>
  <img src='https://github.com/NvsYashwanth/Regression-Master/blob/master/assets/simple%20vs%20poly%20reg.png'>
</p>

* The Simple and Multiple Linear equations are also Polynomial equations with a single degree, and the Polynomial regression equation is Linear equation with the nth degree.
---

# ***4. Learning Curves***
* If a model performs well on the training data but generalizes poorly, then your model is overfitting. If it performs poorly on both, then it is underfitting. This is one way to tell when a model is too simple or too complex.
* Another way to tell is to look at the learning curves: these are plots of the model’s performance on the training set and the validation set as a function of the training set
size (or the training iteration).
```
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

X = 2 * np.random.rand(100, 1)
y = -2 +  3* X + np.random.randn(100, 1)
X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.2)
regressor=LinearRegression()
regressor.fit(X_train,y_train)
predictions=regressor.predict(X_val)

plt.figure(1,figsize=(15,5))
plt.subplot(121)
plt.scatter(X,y)
plt.plot(X_val,predictions,color='black')
plt.title('Scikit Learn Linear Regression')

train_errors=[]
val_errors=[]
plt.subplot(122)
for i in range(1,len(X_train)):
    regressor.fit(X_train[:i],y_train[:i])
    train_preds=regressor.predict(X_train[:i])
    val_preds=regressor.predict(X_val)
    train_errors.append(mse(train_preds,y_train[:i]))
    val_errors.append(mse(val_preds,y_val))
plt.plot(range(1,len(X_train)),np.sqrt(train_errors),label='Train Loss')
plt.plot(range(1,len(X_train)),np.sqrt(val_errors),label='Validation Loss')
plt.title('learning Curves')
plt.xlabel('Train set size')
plt.ylabel('RMSE')
plt.legend()
plt.show()
```
The output of the above code : 
<p align='center'>
  <img src='https://github.com/NvsYashwanth/Regression-Master/blob/master/assets/learning%20curves.png'>
</p>

* When the model is trained for 1 or 2 instances and the same when used for predicting training set gives higher accuracy that is with lower loss but higher validation loss. This is because the model is overfitting the training data and can not generalize in case of validation set. 
* As the number of instances over which the model is being trained increases, the validation loss decreases meaning it starts generalizing better than before. However the training loss reaches a plateau after which adding new instances to the training set doesn’t make the average error much better or worse. 


# ***5. Regularization Models***
* Overfitting happens when model learns signal as well as noise in the training data and wouldn’t perform well on new data on which model wasn’t trained on.
* Few ways you can avoid overfitting is by cross-validation sampling, reducing number of features, pruning, regularization etc.
* Regularization is a good way to avoid overfitting and we shall discuss the same. 
* Regularization basically adds the penalty as model complexity increases. A smiple way to regularize a model is to decrease the number of polynomial dregrees.
* Regularization will add the penalty for higher terms through a regularization parameter lambda. This will decrease the importance given to higher terms thereby decereasing model complexity.
<p align='center'>
  <img src='https://github.com/NvsYashwanth/Regression-Master/blob/master/assets/regularization%20intro.png'>
</p>

* To simply we must define Parsimonious Models. These are models that explain data with a minimum number of parameters, or predictor variables. Thus Regularization means creating a parsimonious model. 
* We will go over three regualarization models which are used to constrain parameters of a linear model.
<p align='center'>
“Everything should be made simple as possible, but not simpler – Albert Einstein”
</p>

## Ridge Regression
* It is also known as Tikhonov regularization.
* Ridge Regression is a L2 regularization technique. In regularization, we reduce the magnitude of the coefficients. Since Ridge Regression shrinks or reduces the number of parameters, it is mostly used to prevent multicollinearity as well as the model complexity.
*
