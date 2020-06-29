# Simple Linear Regression for Grp 1

# y = b0+b1.x1
# y = b + m*x1
# y = depedent variable
# x = independent variable / Matrix of features 
# b0 = constant
# B1 = Slope/ coefficient

# importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# imporrting the data
dataset = pd.read_csv('Salary_Data Simple Linear Regression.csv')

# dividing the data into independent and dependent variable
X = dataset.iloc[:,:-1].values
y = dataset.loc[:,'Salary'].values

# Splitting the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split (X, y,
                                                     test_size = 0.2,
                                                     random_state = 0)
# Regressor
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Prediction
lr.predict([[4.3]])
pred = lr.predict(X_test)

# Visualization (Training dataset)
plt.scatter(X_train, y_train)
plt.plot (X_train, lr.predict(X_train), color = 'red')
plt.title('SLR')
plt.ylabel('Salary')
plt.xlabel('Exp')
plt.show()

# Visualization (Test dataset)
plt.scatter(X_test, y_test)
plt.plot (X_test, lr.predict(X_test), color = 'red')
plt.title('SLR')
plt.ylabel('Salary')
plt.xlabel('Exp')
plt.show()

##### to Save model
import pickle
with open ('SLR', 'wb') as file:
    pickle.dump(lr, file)

with open ('SLR', 'rb') as file:
    sd = pickle.load(file)

y1 = sd.predict([[4.3]])
sd.coef_
sd.intercept_

# y = b + m*x1
39343 = 26780 + (9312*1.1)

# y = mean, y1, y2.....yn = actual values
'''OLS = [(y1-y)**2 + (y2-y)**2 + (y3-y)**2+.......+ (yn-y)**2]
MSE (Mean Sq error) = Cost function'''






