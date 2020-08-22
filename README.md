# Linear Regression
### Linear Regression is a supervised machine learning algorithm that predicts a continuous output.

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

### Cost Function
* We define the cost funciton MSE (Mean Square Error) that measures the average squared differnce between actual and predicted values. Our goal is to minimize this cost function in order to improve the accuracy of the model.

<p align='center'>
  <img src='https://github.com/NvsYashwanth/Sales-prediction/blob/master/assets/MSE.png'>
</p>

* MSE cost function for the Linear Regression model is a convex function, which means that for any 2 points

### Hypothesis
1. There exists a linear relationship between target and feature variables. This can be verified using scatterplots.
2. There is little or no multicollinearity between feature variables. The same can be checked using pariplot and heatmaps.
3. Error terms are normally distributed. Perform residual analysis to verify the same.
