---
layout: post
title: " Mathematical Background of Nonlinear Least Squares Fitting"
date: 2024-10-18
categories: mathematics optimization
---

# Rigorous Mathematical Background of Nonlinear Least Squares Fitting

Nonlinear least squares fitting is a fundamental tool in data analysis, especially in fields like physics, engineering, and applied mathematics. Understanding the rigorous mathematical foundation behind this method is critical for interpreting the results and applying them effectively in real-world problems. 

In this post, we will delve into the **mathematical formulation** of nonlinear least squares, the **Jacobians**, **covariance matrices**, and the **Levenberg-Marquardt algorithm** — one of the most popular methods for solving nonlinear least squares problems.

---

## Mathematical Formulation of Nonlinear Least Squares

In a typical nonlinear least squares problem, we are given a set of data points \((x_i, y_i)\) and a model function \(f(x; \theta)\), where \(\theta\) represents the parameters we wish to estimate. The goal is to find the parameters \(\theta\) that minimize the **sum of squared residuals**:

$$
\Large S(\theta) = \sum_{i=1}^{n} \left( y_i - f(x_i; \theta) \right)^2
$$

Where:
- \( S(\theta) \) is the objective function to minimize.
- \( y_i \) are the observed values.
- \( f(x_i; \theta) \) are the predicted values from the model.
- \( x_i \) are the input values.
- \( \theta \) are the parameters to be fitted.
---

## Jacobian Matrix in Nonlinear Least Squares

To minimize the sum of squared residuals, we need to compute the **gradient** of the objective function. The gradient is derived using the **Jacobian matrix** of the model function \(f(x_i; \theta)\) with respect to the parameters \(\theta\).

The **Jacobian matrix** \(J\) is defined as:

$$
\Large J_{ij} = \frac{\partial f(x_i; \theta)}{\partial \theta_j}
$$

Where:
- \(J\) is an \(n \times m\) matrix, where \(n\) is the number of data points and \(m\) is the number of parameters.
- Each element \(J_{ij}\) is the partial derivative of the model function with respect to the \(j\)-th parameter, evaluated at the \(i\)-th data point.

---

## Covariance Matrix of the Parameters

Once the best-fit parameters \(\hat{\theta}\) are found, we can estimate the **uncertainties** in these parameters using the **covariance matrix**. The covariance matrix is given by:

$$
\Large \Sigma_{\theta} = \sigma^2 (J^T J)^{-1}
$$

Where:
- \(\Sigma_{\theta}\) is the covariance matrix of the parameter estimates.
- \(\sigma^2\) is the variance of the residuals (the squared differences between the observed and predicted values).
- \(J^T\) is the transpose of the Jacobian matrix.
- \(J\) is the Jacobian matrix evaluated at the best-fit parameters.

The **diagonal elements** of the covariance matrix represent the variances of the individual parameters, while the **off-diagonal elements** indicate the covariances between the parameters.

---

## The Levenberg-Marquardt Algorithm

The **Levenberg-Marquardt algorithm** is a widely used method for solving nonlinear least squares problems. It combines the ideas of **gradient descent** and **Gauss-Newton optimization** to provide a robust and efficient algorithm for finding the best-fit parameters.

### Update Rule:

The Levenberg-Marquardt algorithm updates the parameters \(\theta_k\) at iteration \(k\) using the following rule:

$$
\Large \theta_{k+1} = \theta_k - (J^T J + \lambda I)^{-1} J^T \mathbf{r}
$$

Where:
- \(\mathbf{r} = y_i - f(x_i; \theta)\) is the vector of residuals.
- \(J\) is the Jacobian matrix.
- \(\lambda\) is a damping parameter that controls the step size.
- \(I\) is the identity matrix.

The parameter \(\lambda\) is adjusted at each iteration to balance between the **Gauss-Newton method** (when \(\lambda\) is small) and **gradient descent** (when \(\lambda\) is large).

---

## Python Implementation of Nonlinear Least Squares with `curve_fit`

In Python, the **SciPy** library provides the `curve_fit` function, which is based on the **Levenberg-Marquardt algorithm**. Below is an example of how to use `curve_fit` for nonlinear least squares fitting:

```python
from scipy.optimize import curve_fit
import numpy as np

# Define the model function
def model(x, A, B):
    return A * np.exp(B * x)

# Generate synthetic data
xdata = np.linspace(0, 5, 50)
ydata = 3.0 * np.exp(1.5 * xdata) + np.random.normal(size=xdata.size)

# Use curve_fit to fit the model to the data
popt, pcov = curve_fit(model, xdata, ydata)

# popt contains the best-fit parameters A and B
# pcov is the covariance matrix
print("Best-fit parameters:", popt)
print("Covariance matrix:", pcov)
