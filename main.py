# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron

import numpy as np

# Load dataset
filename = "spambase.csv"
dataset = pandas.read_csv(filename,header=None)

# colormap = np.array(['r', 'k'])
# plt.scatter(dataset.loc[:,4], dataset.loc[:,15], c=colormap[dataset.loc[:,57]], s=40)
# plt.show()

# print(dataset.shape) 
# print(dataset.head(20))
# print(dataset.describe())

# Split-out validation dataset
array = dataset.values
X = array[:,0:57] # 0:3 are the features, 4 is the class
Y = array[:,57]  
validation_size = 0.20 
seed = 7
scoring = 'accuracy' # ratio of correct predictions / total nr of instances
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)


print("Training set has {} samples.".format(X_train.shape))
print("Testing set has {} samples.".format(X_validation.shape))

# fit_intercept should be True
# try max_iter = 5000
models = []
models.append(('P1', Perceptron(penalty=None,fit_intercept=True, max_iter=1000)))
# models.append(('P2', Perceptron(penalty='l2', alpha=0.01, fit_intercept=True, max_iter=5000)))
# models.append(('P3', Perceptron(penalty='l1', alpha=0.0001, fit_intercept=True, max_iter=5000)))
# models.append(('P4', Perceptron(penalty='l1', alpha=0.001, fit_intercept=True, max_iter=5000)))
# models.append(('P5', Perceptron(penalty='l1', alpha=0.01, fit_intercept=True, max_iter=5000)))
# models.append(('P6', Perceptron(penalty='l1', alpha=0.1, fit_intercept=True, max_iter=5000)))
# models.append(('P7', Perceptron(penalty='l1', alpha=1, fit_intercept=True, max_iter=5000)))



# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

# # Compare Algorithms
# fig = plt.figure()
# fig.suptitle('Algorithm Comparison')
# ax = fig.add_subplot(111)
# plt.boxplot(results)
# ax.set_xticklabels(names)
# plt.show()

# current best
per = Perceptron(penalty='l1', alpha=0.01, fit_intercept=True, max_iter=1000, tol=None, shuffle=True, verbose=0, eta0=1.0, n_jobs=1, random_state=0, class_weight=None, warm_start=False, n_iter=None)
per.fit(X_train, Y_train)
predictions = per.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
# print(per.score(X_validation, Y_validation))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
#print(per.coef_)
#print(per.n_iter_)
#print(per.intercept_)
