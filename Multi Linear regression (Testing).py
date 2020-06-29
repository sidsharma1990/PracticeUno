# Multi Linear regression (Testing)

# importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Dataset
dataset = pd.read_csv('Regression - Data.csv')

# dividing the data into independent and dependent variable
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

# categorical data
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), 
#                                        [3])], remainder = 'passthrough')
# X = np.array(ct.fit_transform(X))

############# Data Splitting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split (X, y,
                                                     test_size = 0.2,
                                                     random_state = 0)

# Regressor
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Prediction
pred = lr.predict(X_test)

########## Model Evaluation
from sklearn.metrics import r2_score
r2_score(y_test, pred)






