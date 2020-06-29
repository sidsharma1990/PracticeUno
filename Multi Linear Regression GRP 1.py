# Multi Linear regression

# importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Dataset
dataset = pd.read_csv('50_Startups.csv')

# dividing the data into independent and dependent variable
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

# categorical data
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), 
                                       [3])], remainder = 'passthrough')
X = np.array(ct.fit_transform(X))

############# Data Splitting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split (X, y,
                                                     test_size = 0.2,
                                                     random_state = None)

# Regressor
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Prediction
pred = lr.predict(X_test)

########## Selection or Elimination
import statsmodels.api as sm
X1 = np.array(X[:, [0,1,2,3,4,5]], dtype = float)
ols = sm.OLS(endog = y, exog = X1).fit()
ols.summary()

X1 = np.array(X[:, [0,1,2,4,5]], dtype = float)
ols = sm.OLS(endog = y, exog = X1).fit()
ols.summary()

X1 = np.array(X[:, [1,2,4,5]], dtype = float)
ols = sm.OLS(endog = y, exog = X1).fit()
ols.summary()

X1 = np.array(X[:, [1,2,4]], dtype = float)
ols = sm.OLS(endog = y, exog = X1).fit()
ols.summary()









