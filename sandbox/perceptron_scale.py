# Load libraries
import pandas
from sklearn import preprocessing
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC

import numpy as np

# Load dataset
filename = "spambase.csv"
dataset = pandas.read_csv(filename,header=None)

# Split-out validation dataset
array = dataset.values
X = array[:,0:57] # 0:3 are the features, 4 is the class
Y = array[:,57]  
validation_size = 0.20 
seed = 7
scoring = 'accuracy' # ratio of correct predictions / total nr of instances
X_tra, X_tes, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

scaler=preprocessing.StandardScaler().fit(X_tra)


X_train = scaler.transform(X_tra)
X_test = scaler.transform(X_tes)

k_cross = 10
params = {'alpha': [0,	0.00001,	0.0001,	0.001,	0.01,	0.1,	1]}# , 0.001, 0.01]}
#params = {'alpha': [0.0001, 0.001]}



mod = Perceptron(penalty='l1',random_state=0, max_iter=5000)

clf = model_selection.GridSearchCV(mod, param_grid=params, cv=k_cross)
clf.fit(X_train, Y_train)

best_clf = clf.best_estimator_ # print
cross_val_score = clf.best_score_ # print
cv_results = clf.cv_results_ # print

best_clf.fit(X_train, Y_train)

yhat = best_clf.predict(X_test) # print
test_err = best_clf.score(X_test, Y_test) # print

y = best_clf.predict(X_train) # print
train_err = best_clf.score(X_train, Y_train) # print. zero if linearly separable?

print('The best model for the perceptron is')
print(best_clf)
print('It has the best cross-validation error/score, which is')
print(cross_val_score)
print('The entire cv_results: ')
print(cv_results)

print('After training on the whole training data, the training error is')
print(train_err)
print('And the test error is')
print(test_err)