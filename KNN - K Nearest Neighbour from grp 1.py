# KNN - K Nearest Neighbour 
# Libraries
import pandas as pd
import numpy as np

dataset = pd.read_csv('Car Churn.csv')

X = dataset.iloc[:,[1,2]].values
y = dataset.iloc[:,-1].values

# data splitting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2,
                                                    random_state = None)

# Standardization
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Model
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier()
classifier.fit(X_train, y_train)

Pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, Pred)
accuracy_score(y_test, Pred)


# (Cross velidation)

