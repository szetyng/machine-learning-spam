# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron

# Load dataset
filename = "spambase.csv"
dataset = pandas.read_csv(filename,header=None)

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
print(dataset.describe())
# 3,54,55,56

plt.plot(X_train[:,54],X_train[:,55], 'ro')
plt.ylabel('some numbers')
plt.show()