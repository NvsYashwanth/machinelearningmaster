# Sales Prediction - Linear Regression
Linear Regression is a supervised machine learning algorithm that predicts a continuous output.


## Simple Regression
* We define the linear relationship for a given dependent variable `y` as ouput and the independent variable `x`as input. The model tries to fit a straight line by predicting the coefficients `m`(slope) and `b`(intercept) where variables `m` and `b` are optimized to produce accurate predictions.
<p align='center'>
  <img src="https://github.com/NvsYashwanth/Sales-prediction/blob/master/assets/simple%20regression.png">
</p>

## Cost Function
* We define the cost funciton MSE (Mean Square Error) that measures the average squared differnce between actual and predicted values. Our goal is to minimize this cost function in order to improve the accuracy of the model.

<p align='center'>
  <img src='https://github.com/NvsYashwanth/Sales-prediction/blob/master/assets/MSE.png'>
</p>


## Hypothesis
1. There exists a linear relationship between target and feature variables. This can be verified using scatterplots.
2. There is little or no multicollinearity between feature variables. The same can be checked using pariplot and heatmaps.
3. Error terms are normally distributed. Perform residual analysis to verify the same.

## Predictions
### Scikit Learn
* With scikit learn, the coefficients are obtained as :
<p align='center'>
  y=0.05473199*x + 7.14382225
</p>

* Mean Absolute Error: 1.8639438916550555
* Mean Squared Error: 5.569539193467241
* Root Mean Squared Error: 2.3599871172248466

### Statsmodel
* 

