# Polynomial grp 1_2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_excel('Age polynomial regression.xlsx')
X = dataset.iloc[:,:-1].values
y = dataset.loc[:, 'height'].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split (X, y,
                                                     test_size = 0.2)

# Linear Regressor
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Linear regression Visualization
plt.scatter(X_train, y_train, color = 'green')
plt.plot (X_train, lr.predict(X_train), color = 'red')
plt.title('Linear regression')
plt.ylabel('Salary')
plt.xlabel('Exp')
plt.show()

# Polynomial feature
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X_train)

# Integration
lr2 = LinearRegression()
lr2.fit(X_poly, y_train)

# Polynomial regression Visualization
plt.scatter(X_train, y_train)
plt.plot(X_train, lr2.predict(poly_reg.fit_transform(X_train)), color = 'green')
plt.title('Polynomial regression')
plt.ylabel('Salary')
plt.xlabel('Exp')
plt.show()

lr2.predict(poly_reg.fit_transform([[24]]))
lr2.predict(poly_reg.fit_transform([[30]]))

lr.predict([[24]])

pred = lr2.predict(poly_reg.fit_transform(X_test))

plt.scatter(X_test, y_test)
plt.plot(X_test, lr2.predict(poly_reg.fit_transform(X_test)), color = 'green')
plt.title('Polynomial regression')
plt.ylabel('Salary')
plt.xlabel('Exp')
plt.show()
































