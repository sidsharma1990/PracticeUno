# Polynomial Regression
# overfitting (High Variance) = Low error in Training set and High rate of error in Test set
# underfitting (High bias and high variance) = Training and test both are giving high rate of error

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:, -1].values

# Linear Regressor
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X, y)

# Polynomial feature
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 5)
X_poly = poly_reg.fit_transform(X)

# Integration
lr2 = LinearRegression()
lr2.fit(X_poly, y)

# Linear regression Visualization
plt.scatter(X, y, color = 'green')
plt.plot (X, lr.predict(X), color = 'red')
plt.title('Linear regression')
plt.ylabel('Salary')
plt.xlabel('Exp')
plt.show()

# lr.predict([[7]])

# Polynomial regression Visualization
plt.scatter(X,y)
plt.plot(X, lr2.predict(poly_reg.fit_transform(X)), color = 'green')
plt.title('Polynomial regression')
plt.ylabel('Salary')
plt.xlabel('Exp')
plt.show()

lr2.predict(poly_reg.fit_transform([[7]]))












