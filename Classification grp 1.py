# Classification
# Logistic Regression - Classifications's Linear Regression

# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_excel('Logistic regression Home.xlsx')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# plotting
plt.scatter(dataset.age, dataset.bought_home)

# Data Splitting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split (X,y, test_size = 0.2,
                                                     random_state = None)


from sklearn.linear_model import LogisticRegression
classfier = LogisticRegression(random_state=0)
classfier.fit(X_train, y_train)

pred = classfier.predict(X_test)
prob=classfier.predict_proba(X_test)

log_prob=classfier.predict_log_proba(X_test)


#################### 2 dataset
dataset = pd.read_excel('Classified Data.xlsx')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split (X,y, test_size = 0.2,
                                                     random_state = None)

from sklearn.linear_model import LogisticRegression
classfier = LogisticRegression(random_state=0)
classfier.fit(X_train, y_train)

pred = classfier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, pred)
accuracy_score(y_test, pred) # 96.5
print (cm)

# Confusion Matrix
[[105   0] = 1 = 100% accurate (yes)
 [  7  88]] = 0 = 92.631












