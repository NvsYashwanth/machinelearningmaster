---
layout: page
title: 6.2. Beyond First Order Optimization
permalink: /beyond-first-order-optimization/
parent: 6. Understanding Gradient Descent
nav_order: 2
---
# ***6.2. Beyond First Order Optimization***

* As mentioned earlier, Gradient Descent is a first-order optimization algorithm meaning it only measures the slope of the cost function curve but not the curvature. ( Curvature means the degree by which a curve or a surface deviates from being a straight line or a plane respectively).

* So how nature of the function or its curvature measured? It is determined by the second-order derivative. The curvature affects our training.

* ***If the Second-order derivative is equal to 0***, the curvature is said to be linear.

* ***If the Second-order derivative is greater than 0***, the curvature is said to be moving upward.

* ***If the Second-order derivative is less than 0***, the curvature is said to be moving downwards.