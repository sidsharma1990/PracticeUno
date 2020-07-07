# Decision Tree testing

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

from sklearn.tree import DecisionTreeRegressor
reg = DecisionTreeRegressor(random_state = 0)
reg.fit(X_train,y_train)

y_pred = reg.predict(X_test)


########## Model Evaluation
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)

92.26




















