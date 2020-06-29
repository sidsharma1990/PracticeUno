# Data Preprocessing template GRP 1

# importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# imporrting the data
dataset = pd.read_csv('Data.csv')

# deviding the data into independent and dependent variable
X = dataset.iloc[:,:-1].values
y = dataset.loc[:,'Purchased'].values

# Filling missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

# Splitting the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split (X, y,
                                                     test_size = 0.2)








