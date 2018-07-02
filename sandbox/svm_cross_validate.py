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
test_sz = 0.20 
seed = 7
scoring = 'accuracy' # ratio of correct predictions / total nr of instances
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_sz, random_state=seed)

scoring = ['precision_macro', 'recall_macro']
clf = SVC(C=1, random_state=0)
scores = model_selection.cross_validate(clf, X_train, Y_train, scoring=scoring, cv=5, return_train_score=True)
print('keys: ' + str(scores.keys()))
print ('test recall_macro: ' + str(scores['test_recall_macro'])) # test_recall_macro is one of the keys
print ('train recall_macro: ' + str(scores['train_recall_macro']))
print('test precision_macro: ' + str(scores['test_precision_macro']))
print('train precision_macro: ' + str(scores['train_precision_macro']))
print('fit_time: ' + str(scores['fit_time']))
print('fit_time: ' + str(scores['fit_time']))
print('score_time: ' + str(scores['score_time']))

