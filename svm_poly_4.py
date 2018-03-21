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
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# print("Training set has {} samples.".format(X_train.shape))
# print("Testing set has {} samples.".format(X_validation.shape))

# fit_intercept should be True
# try max_iter = 5000
# evaluate each model in turn
models = []
val_errors = []
names = []
cur_err = 0
cur_clf = SVC()

gammas = [0.00001, 0.0001, 0.001]
cvals = [1,10,100,1000]

i = 0
for cval in cvals:
	for gamma in gammas:
		models.append(('SVC with C: '+str(cval)+',gamma: '+str(gamma), SVC(C=cval,cache_size=1000,gamma=gamma,random_state=0,kernel='poly',degree=4)))
		i += 1

for name, model in models:
	kfold = 10
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	cv_err = cv_results.mean()
	val_errors.append(cv_results)
	#names.append(name)
	msg = "%s: %f (%f)" % (name, cv_err, cv_results.std())
	print(msg)
	if cv_err > cur_err:
		cur_err = cv_err
		cur_clf = model

cur_clf.fit(X_train, Y_train)
predictions = cur_clf.predict(X_validation)
print('Best model is:')
print(cur_clf)
print('With a test score of: ' + str(accuracy_score(Y_validation, predictions)))
ytrained = cur_clf.predict(X_train)
print('Training score: ' + str(accuracy_score(Y_train, ytrained)))
# print('Took ' + str(cur_clf.n_iter_) + ' iterations')
# print(confusion_matrix(Y_validation, predictions))
# print(classification_report(Y_validation, predictions))





# current best
# per = Perceptron(penalty='l1', alpha=0.01, fit_intercept=True, max_iter=1000, tol=None, shuffle=True, verbose=0, eta0=1.0, n_jobs=1, random_state=0, class_weight=None, warm_start=False, n_iter=None)
# per.fit(X_train, Y_train)
# predictions = per.predict(X_validation)
# print(accuracy_score(Y_validation, predictions))
# # print(per.score(X_validation, Y_validation))
# print(confusion_matrix(Y_validation, predictions))
# print(classification_report(Y_validation, predictions))
#print(per.coef_)
#print(per.n_iter_)
#print(per.intercept_)
