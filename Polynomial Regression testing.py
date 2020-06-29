# Polynomial Regression testing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Regression - Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, -1].values

corr1 = dataset.corr()

############# Data Splitting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split (X, y,
                                                     test_size = 0.2,
                                                     random_state = 0)

# Linear Regressor
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Polynomial feature
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 5)
X_poly = poly_reg.fit_transform(X_train)

# Integration
lr2 = LinearRegression()
lr2.fit(X_poly, y_train)

pred = lr2.predict(poly_reg.fit_transform(X_test))

########## Model Evaluation
from sklearn.metrics import r2_score
r2_score(y_test, pred)

94.478




















