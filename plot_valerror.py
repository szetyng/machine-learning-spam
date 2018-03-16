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



models = []
val_errors = []
names = []
cur_err = 0
cur_clf = MLPClassifier(random_state=0)

alphas = [0, 0.0001, 0.001, 0.01, 0.1, 1, 10]
act = 'tanh'
# alphas = [10, 100, 1000, 5000]
# reg = ['l2']



# i = 0
# for alp in alphas:
# 	models.append(('NN'+str(i)+' '+a, MLPClassifier(random_state=0,activation=a,alpha=alp)))
# 	i += 1

# for name, model in models:
# 	kfold = model_selection.KFold(n_splits=10, random_state=seed)
# 	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
# 	cv_err = cv_results.mean()
# 	val_errors.append(cv_results)
# 	#names.append(name)
# 	msg = "%s: %f (%f)" % (name, cv_err, cv_results.std())
# 	print(msg)
# 	if cv_err > cur_err:
# 		cur_err = cv_err
# 		cur_clf = model

# cur_clf.fit(X_train, Y_train)
# predictions = cur_clf.predict(X_validation)
# print('Best model is:')
# print(cur_clf)
# print('With an accuracy score of: ' + str(accuracy_score(Y_validation, predictions)))
# print(confusion_matrix(Y_validation, predictions))
# print(classification_report(Y_validation, predictions))
# print('Nr of iterations: ' + str(cur_clf.n_iter_))

X, y = X_train, Y_train
# print(X.shape)
# print(y.shape)
param_range = [0, 0.0001, 0.001, 0.01, 0.1]
train_scores, test_scores = model_selection.validation_curve(
    MLPClassifier(random_state=0,activation='tanh'), X, y, param_name="alpha", param_range=param_range,cv=10, scoring="accuracy", n_jobs=1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve with SVM")
plt.xlabel("$\gamma$")
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
lw = 2
plt.semilogx(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.show()

