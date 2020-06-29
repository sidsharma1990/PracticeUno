# Data preprocessing Grp 1 
# categorical data - Info except number
# x is features of matrix

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.loc[:,'Purchased'].values
# y1 = dataset.iloc[:, -1].values
# y2 = dataset.iloc[:, 3].values
X1 = dataset.iloc[:,:-1].values

# pandas
# dataset['Age'].fillna(value = dataset['Age'].mean(), inplace = True)
# dataset['Salary'] = dataset['Salary'].fillna(dataset['Salary'].median())
# df2 = dataset.fillna({'Age':dataset['Age'].mean(),
#                       'Salary': dataset['Salary'].median()})
# dataset['Salary'] = dataset['Salary'].fillna(method = 'ffill')

# Scikit - Imputer
# fit and Transform
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

# fit and transform together
# from sklearn.impute import SimpleImputer
# imputer = SimpleImputer(missing_values=np.nan, strategy='median')
# X[:,1:3] = imputer.fit_transform(X[:,1:3])

# correlation
dataset[['Age', 'Salary']].corr()

#### Pandas
# dummy = pd.get_dummies(dataset['State'])
# X1 = pd.concat([X1, dummy], axis = 1)
# X1.drop(['State', 'Mumbai'], axis = 1, inplace = True)

# Transforming the data
# columnstransformer, onehot encoder

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), 
                                       [0])], remainder = 'passthrough')
X = np.array(ct.fit_transform(X))

# Dependent
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

############# Data Splitting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split (X, y,
                                                     test_size = 0.2)

# Data Scaling (-3 to +3)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])

# normalizer
from sklearn.preprocessing import Normalizer
nz = Normalizer()
X_train[:, 3:] = nz.fit_transform(X_train[:, 3:])
X_test[:, 3:] = nz.transform(X_test[:, 3:])











