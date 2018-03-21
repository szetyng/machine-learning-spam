# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.neural_network import MLPClassifier

import numpy as np

######################## Data preparation #############################
# Load dataset
filename = "spambase.csv"
dataset = pandas.read_csv(filename,header=None)

# Split-out validation dataset
array = dataset.values
X = array[:,0:57] # 0:3 are the features, 4 is the class
Y = array[:,57]  
test_sz = 0.20 
seed = 7
scoring = 'accuracy' # ratio of correct predictions / total nr of instances
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_sz, random_state=seed)

# print(dataset.shape)
# np.histogram(X_train[:,0])
# plt.hist(X_train)
# plt.show()


#################### Cross-validation #################################

k_cross = 10
params = [	{'gamma': [0.00001, 0.0001, 0.001], 'C': [10,100,1000], 'kernel': ['rbf']},
			{'gamma': [0.00001, 0.0001, 0.001], 'C':[10,100,1000], 'kernel': ['poly']} ]
# params = {'gamma': [0.0001, 0.001], 'C': [1,10,100], 'kernel': ['rbf']}

clf = model_selection.GridSearchCV(SVC(cache_size=1000,random_state=0), param_grid=params,refit=True, cv=k_cross)
clf.fit(X_train, Y_train)

best_clf = clf.best_estimator_ # print
cross_val_score = clf.best_score_ # print
cv_results = clf.cv_results_ # print

best_clf.fit(X_train, Y_train)

yhat = best_clf.predict(X_test) # print
test_err = best_clf.score(X_test, Y_test) # print

y = best_clf.predict(X_train) # print
train_err = best_clf.score(X_train, Y_train) # print. zero if linearly separable?

print('The best model for SVM is')
print(best_clf)
print('It has the best cross-validation error/score, which is')
print(cross_val_score)
print('The entire cv_results: ')
print(cv_results)

print('After training on the whole training data, the training error is')
print(train_err)
print('And the test error is')
print(test_err)
# print('Just for fun, the predictions are')
# print(yhat)
# print('And for training data, the predictions are')
# print(y)

print('done')
