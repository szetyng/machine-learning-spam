import numpy as np
import matplotlib.pyplot as plt
import pandas
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import preprocessing

heldout = [0.95, 0.90, 0.75, 0.50, 0.01]
rounds = 20

# Load dataset
filename = "spambase.csv"
dataset = pandas.read_csv(filename,header=None)

# Split-out validation dataset
array = dataset.values
X_train = array[:,0:57] # 0:3 are the features, 4 is the class
y = array[:,57]  

scaler=preprocessing.StandardScaler().fit(X_train)

X = scaler.transform(X_train)



classifiers = [
    # ("SVM", SVC(C=100000, cache_size=2000,gamma=0.0001, kernel='rbf',random_state=0)),
    ("Perceptron",Perceptron(alpha=0.0001,max_iter=5000, penalty='l1', random_state=0,))
]

xx = 1. - np.array(heldout)
xx_train = 1. - np.array(heldout)

for name, clf in classifiers:
    print("training %s" % name)
    rng = 0
    yy = []
    yy_train = []
    for i in heldout:
        yy_ = []
        yy_train_ = []
        for r in range(rounds):
            X_train, X_test, y_train, y_test = \
                train_test_split(X, y, test_size=i, random_state=rng)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_mytrain = clf.predict(X_train)
            yy_.append(1 - np.mean(y_pred == y_test))
            yy_train_.append(1 - np.mean(y_mytrain == y_train))
        yy.append(np.mean(yy_))
        yy_train.append(np.mean(yy_train_))
    plt.plot(xx, yy, 'r',label=name+' test curve')
    plt.plot(xx_train,yy_train,'b',label=name+' training curve')

plt.legend(loc="upper right")
plt.xlabel("Proportion of data")
plt.ylabel("Error Rate")
plt.show()