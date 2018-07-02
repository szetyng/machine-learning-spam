# Load libraries
import pandas
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn import preprocessing
from time import time
import numpy as np
import pylab

# Load dataset
filename = "..\data\spambase.csv"
dataset = pandas.read_csv(filename,header=None)

# Split test dataset
array = dataset.values
X = array[:,0:57] # 0:3 are the features, 4 is the class
Y = array[:,57]  
validation_size = 0.20 
seed = 7
scoring = 'accuracy' # ratio of correct predictions / total nr of instances
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Scale data for SVM
scaler=preprocessing.StandardScaler().fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Cross-validation for perceptron
k_cross = 10
params = {'alpha': [0.00001, 0.0001,0.001,0.01,0.1,1]}
mod = Perceptron(penalty='l1',random_state=0, max_iter=5000)
clf = model_selection.GridSearchCV(mod, param_grid=params, cv=k_cross)
clf.fit(X_train, Y_train)

best_clf = clf.best_estimator_ 
cross_val_score = clf.best_score_ 
cv_results = clf.cv_results_ 

best_clf.fit(X_train, Y_train)

yhat = best_clf.predict(X_test) 
test_err = best_clf.score(X_test, Y_test) 

y = best_clf.predict(X_train) 
train_err = best_clf.score(X_train, Y_train) 

print('The best model for the perceptron is')
print(best_clf)
print('It has the best cross-validation error/score, which is')
print(cross_val_score)

print('After training on the whole training data, the training error is')
print(train_err)
print('And the test error is')
print(test_err)

# Cross validation for SVM
params = {'gamma': [1e-8,1e-7,1e-6,1e-5,1e-4,0.001,0.01], 'C': [1000,1e4,1e5,1e6,1e7]}

clf = model_selection.GridSearchCV(SVC(cache_size=2000,random_state=0,kernel='rbf'), param_grid=params,refit=True, cv=k_cross)
clf.fit(X_train_scaled, Y_train)

best_clf = clf.best_estimator_ # print
cross_val_score = clf.best_score_ # print
cv_results = clf.cv_results_ # print

best_clf.fit(X_train_scaled, Y_train)

yhat = best_clf.predict(X_test_scaled) # print
test_err = best_clf.score(X_test_scaled, Y_test) # print

y = best_clf.predict(X_train_scaled) # print
train_err = best_clf.score(X_train_scaled, Y_train) 

print('The best model for SVM is')
print(best_clf)
print('It has the best cross-validation error/score, which is')
print(cross_val_score)

print('After training on the whole training data, the training error is')
print(train_err)
print('And the test error is')
print(test_err)

# plot graphs
# Perceptron graph
y_val = [0.84456522,	0.89021739,	0.90461957,	0.89157609,	0.79293478,	0.71684783,	0.64755435]
y_line=[]
for i in y_val:
	y_line.append(1-i)

x_line = [0,	0.00001,	0.0001,	0.001,	0.01,	0.1,	1]

plt.plot(x_line,y_line)
plt.xscale('log')
plt.xlabel('alpha')
plt.ylabel('Cross-validation error')
plt.title('Perceptron: Cross-validation error against alpha')
plt.show()

# SVC scaled
# for c = 100000
y_val = [0.899722,	0.922594,	0.927988,	0.934235,	0.937498,	0.924723,	0.898905]
y_line=[]
for i in y_val:
	y_line.append(1-i)
x_line = [1e-8,	1e-7,	1e-6,	1e-5,	0.0001,	0.001,	0.01]

plt.plot(x_line,y_line)
plt.xscale('log')
plt.xlabel('gamma')
plt.ylabel('Cross-validation error')
plt.title('SVM: Cross-validation error against gamma for C = 100,000')
plt.show()

# for gamma=0.0001
y_val = [0.921196,0.931519,0.937498,0.934510,0.932877]
y_line=[]
for i in y_val:
	y_line.append(1-i)
x_line = [10000000,1000000,100000,10000,1000]

plt.plot(x_line,y_line)
plt.xscale('log')
plt.xlabel('C')
plt.ylabel('Cross-validation error')
plt.title('SVM: Cross-validation error against C for gamma = 0.0001')
plt.show()
