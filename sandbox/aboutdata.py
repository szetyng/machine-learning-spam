from sklearn import preprocessing
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from time import time

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

X_train_data = pandas.DataFrame(X_train)
print(str(X_train_data.describe()))

# scaler=preprocessing.StandardScaler().fit(X_train)

# X_train_scaled = scaler.transform(X_train)
# X_train_scaled_data = pandas.DataFrame(X_train_scaled)
# print(X_train_scaled_data.describe())