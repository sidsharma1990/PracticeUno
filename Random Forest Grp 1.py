# Random Forest 
''' Also called as esemble technique'''
# basically under bagging category

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor(n_estimators=10, random_state = 0)
reg.fit(X,y)

# Prediction
reg.predict([[6.5]])








