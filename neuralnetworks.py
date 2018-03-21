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
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# models = []
# # models.append(('SVM', SVC(C=3, shrinking=False)))
# models.append(('NN1', MLPClassifier(random_state=0)))
# models.append(('NN2', MLPClassifier(random_state=0,hidden_layer_sizes=(200,))))
# models.append(('NN3', MLPClassifier(random_state=0,activation='logistic')))
# models.append(('NN4', MLPClassifier(random_state=0,activation='tanh')))
# models.append(('NN5', MLPClassifier(random_state=0,alpha=0.001)))


# # evaluate each model in turn
# results = []
# names = []
# for name, model in models:
# 	kfold = model_selection.KFold(n_splits=10, random_state=seed)
# 	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
# 	results.append(cv_results)
# 	names.append(name)
# 	msg = "Validation error for %s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
# 	print(msg)
# 	model.fit(X_train, Y_train)
# 	predictions = model.predict(X_validation)
# 	print('Test error is ' + str(accuracy_score(Y_validation, predictions)))
# 	print(confusion_matrix(Y_validation, predictions))
# 	print(classification_report(Y_validation, predictions))





# HERE


models = []
val_errors = []
names = []
cur_err = 0
cur_clf = MLPClassifier(random_state=0)

alphas = [0, 0.0001, 0.001, 0.01, 0.1, 1, 10]
act = ['identity', 'logistic', 'tanh', 'relu']
# alphas = [10, 100, 1000, 5000]
# reg = ['l2']

i = 0
for a in act:
	for alp in alphas:
		models.append(('NN'+str(i)+' '+a, MLPClassifier(random_state=0,activation=a,alpha=alp)))
		i += 1

for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
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
print('With an accuracy score of: ' + str(accuracy_score(Y_validation, predictions)))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
print('Nr of iterations: ' + str(cur_clf.n_iter_))
