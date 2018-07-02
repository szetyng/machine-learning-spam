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



# Printing training errors

clf1 = Perceptron(penalty='l1',random_state=0, max_iter=5000, alpha=0.0001)
clf2 = Perceptron(penalty='l1',random_state=1, max_iter=5000, alpha=0.0001)
clf3 = Perceptron(penalty='l1',random_state=2, max_iter=5000, alpha=0.0001)
clf4 = Perceptron(penalty='l1',random_state=3, max_iter=5000, alpha=0.0001)

clf1.fit(X_train,Y_train)
p1 = clf1.predict(X_train)
err1 = clf1.score(X_train, Y_train)

clf2.fit(X_train,Y_train)
p2 = clf2.predict(X_train)
err2 = clf2.score(X_train, Y_train)

clf3.fit(X_train,Y_train)
p3 = clf3.predict(X_train)
err3 = clf3.score(X_train, Y_train)

clf4.fit(X_train,Y_train)
p4 = clf4.predict(X_train)
err4 = clf4.score(X_train, Y_train)

print(err1)
print(err2)
print(err3)
print(err4)


