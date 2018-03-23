- [Perceptron](#perceptron)
    - [Cross-validation results](#cross-validation-results)
    - [Performance](#performance)
- [SVM - RBF kernel](#svm---rbf-kernel)
    - [Cross-validation results](#cross-validation-results)
    - [Performance](#performance)
- [SVM RBF scaled](#svm-rbf-scaled)
    - [Cross-validation errors](#cross-validation-errors)
    - [Performance](#performance)

# Perceptron
## Cross-validation results 
|Alpha      |0          |0.00001    |0.0001     |0.001      |0.01       |0.1        |1          |
|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
|CV errror  |0.84456522 |0.89021739 |**0.90461957** |0.89157609 |0.79293478 |0.71684783 |0.64755435 |

Best model: 
    alpha = 0.0001

```
Perceptron(alpha=0.0001, class_weight=None, eta0=1.0, fit_intercept=True,
      max_iter=5000, n_iter=None, n_jobs=1, penalty='l1', random_state=0,
      shuffle=True, tol=None, verbose=0, warm_start=False)
```

## Performance
For alpha = 0.0001:

|Training error     |Cross-val error    | Test error    |
|-------------------|-------------------|---------------|
|0.8940217391304348 |0.90461957         |0.8805646036916395|

For alpha = 0.0001, the training errors for different random states:

|Random state           |0                      |1                      |2                      |3          |
|---|---|---|---|---|
|                       |0.8940217391304348     |0.8983695652173913     |0.9241847826086956     |0.9125     |

Confusion matrix:
[[502  47]
 [ 69 303]]

Classification report:
             precision    recall  f1-score   support

        0.0       0.88      0.91      0.90       549
        1.0       0.87      0.81      0.84       372

avg / total       0.87      0.87      0.87       921

Fit time: 6.227154731750488 s

# SVM - RBF kernel
## Cross-validation results
|C\Gamma    |1e-08      |1e-07       |0.000001   |0.00001    |0.0001     |0.001      |
|---|---|---|---|---|---|---|
|10000000   |0.914397   |0.928257   |0.932602   |0.914394   |0.893753   |0.860611   |
|1000 000    |0.917119   |0.932058   |**0.938855**|0.924722   |0.908963   |0.866043   |
|100000     |0.883692   |0.922279   |0.934780   |0.933695   |0.912769   |0.875008   |
|10000      |0.775004   |0.886137   |0.926083   |0.931519   |0.917659   |0.886146   |
|1000       |0.735058   |0.785334   |0.89048913 |0.92065217 |0.91902174 |0.89592391 |
|100        |0.711684   |0.732884   |0.78423913 |0.87961957 |0.90842391 |0.90081522 |
|10         |0.679075   |0.719831   |0.73423913 |0.76222826 |0.85923913 |0.87690217 |
|1          |0.658423   |0.680428   |0.705972   |0.727993   |0.757069   |0.809503   |

Best model:
    C = 1 000 000
    gamma = 1e-06

```
SVC(C=100000, cache_size=2000, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=1e-06, kernel='rbf',
  max_iter=-1, probability=False, random_state=0, shrinking=True,
  tol=0.001, verbose=False)
```

## Performance
For C = 1 000 000, gamma = 1e-06

|Training error     |Cross-val error    |Test error         |
|---|---|---|
|0.9548913043478261 |0.938855           |0.9163952225841476 |

There are [308 310] support vectors  
Fit time for non-scaled RBF SVM is: 109.71337223052979

# SVM RBF scaled
## Cross-validation errors
|C\gamma    |1e-8       |1e-7       |1e-6       |1e-5       |0.0001     |0.001      |0.01       |       
|---|---|---|---|---|---|---|---|
|10000000   |0.907875   |0.918201   |0.916299   |0.931786   |0.921196   |0.903256   |0.889673   |
|1000000    |0.914939   |0.913580   |0.921465   |0.935055   |0.931519   |0.916844   |0.898095   |
|100000     |0.899722   |0.922594   |0.927988   |0.934235   |**0.937498**   |0.924723   |0.898905   |
|10000      |0.811679   |0.905430   |0.922277   |0.932606   |0.934510   |0.931517   |0.915482   |
|1000       |0.608424   |0.814669   |0.906246   |0.924179   |0.932877   |0.936954   |0.926079   |

Best model is:
    C = 100000
    gamma = 0.0001

```
SVC(C=100000, cache_size=2000, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.0001, kernel='rbf',
  max_iter=-1, probability=False, random_state=0, shrinking=True,
  tol=0.001, verbose=False)
```

## Performance
For C = 100000, gamma = 0.0001

|Training error     |Cross-val error    |Test error         |
|---|---|---|
|0.9611413043478261 |0.937498           |0.9381107491856677 |

There are [306 301] support vectors, for each class.  
The support vectors are of this shape (607, 57):

Confusion matrix:
[[525  24]
 [ 33 339]]

Classification report:
             precision    recall  f1-score   support

        0.0       0.94      0.96      0.95       549
        1.0       0.93      0.91      0.92       372

avg / total       0.94      0.94      0.94       921

Fit time for scaled RBF SVM is: 2.790419578552246
