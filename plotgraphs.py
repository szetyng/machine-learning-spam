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
import pylab
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


# Perceptron graph

y_line = [0.84456522,	0.89021739,	0.90461957,	0.89157609,	0.79293478,	0.71684783,	0.64755435]
x_line = [0,	0.00001,	0.0001,	0.001,	0.01,	0.1,	1]

plt.plot(x_line,y_line)
plt.xscale('log')
plt.xlabel('alpha')
plt.ylabel('Cross-validation error')
plt.title('Perceptron: Cross-validation error against alpha')
plt.show()

# SVC graph
# for c = 1mil

# y_line = [0.917119,	0.932058,	0.938855,	0.924722,	0.908963,	0.866043]
# x_line = [1e-08,	1e-07,	0.000001,	0.00001,	0.0001,	0.001]

# plt.plot(x_line,y_line)
# plt.xscale('log')
# plt.xlabel('gamma')
# plt.ylabel('Cross-validation error')
# plt.title('SVM: Cross-validation error against gamma for C = 1e06')
# plt.show()

# for gamma=1micro
# y_line = [0.932602,0.938855,0.934780,0.926083,0.89048913,0.78423913,0.73423913,0.705972]
# x_line = [10000000,	1000000,100000,10000,1000,100,10,1]

# plt.plot(x_line,y_line)
# plt.xscale('log')
# plt.xlabel('C')
# plt.ylabel('Cross-validation error')
# plt.title('SVM: Cross-validation error against C for gamma = 1e-06')
# plt.show()

# SVC scaled
# for c = 100000
# y_line = [0.899722,	0.922594,	0.927988,	0.934235,	0.937498,	0.924723,	0.898905]
# x_line = [1e-8,	1e-7,	1e-6,	1e-5,	0.0001,	0.001,	0.01]

# plt.plot(x_line,y_line)
# plt.xscale('log')
# plt.xlabel('gamma')
# plt.ylabel('Cross-validation error')
# plt.title('SVM: Cross-validation error against gamma for C = 100,000')
# plt.show()

# for gamma=0.0001
# y_line = [0.921196,0.931519,0.937498,0.934510,0.932877]
# x_line = [10000000,1000000,100000,10000,1000]

# plt.plot(x_line,y_line)
# plt.xscale('log')
# plt.xlabel('C')
# plt.ylabel('Cross-validation error')
# plt.title('SVM: Cross-validation error against C for gamma = 0.0001')
# plt.show()