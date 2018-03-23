- [Perceptron:](#perceptron)
      - [Table for perceptron](#table-for-perceptron)
      - [Perceptron 1](#perceptron-1)
      - [Perceptron 2](#perceptron-2)
- [SVM](#svm)
      - [Table for RBF SVM](#table-for-rbf-svm)
      - [RBF](#rbf)
- [Raw data](#raw-data)
      - [SVM polynomial degree 2](#svm-polynomial-degree-2)
      - [SVM rbf](#svm-rbf)
      - [SVM RBF scaled](#svm-rbf-scaled)
      - [Perceptron scaled](#perceptron-scaled)
# Perceptron:
## Table for perceptron

|Alpha      |0          |0.00001    |0.0001     |0.001      |0.01       |0.1        |1          |
|-------|---|-----------|-------|-------|-------|-------|---|
|CV errror  |0.84456522 |0.89021739 |0.90461957 |0.89157609 |0.79293478 |0.71684783 |0.64755435 |

For alpha = 0.0001:
|Training error     |Cross-val error    | Test error    |
|---|---|---|
|0.8940217391304348 |0.90461957         |0.8805646036916395|

For alpha = 0.0001, the training errors for different random states:
|Random state           |0                      |1                      |2                      |3          |
|---|---|---|---|---|---|
|                       |0.8940217391304348     |0.8983695652173913     |0.9241847826086956     |0.9125     |

## Perceptron 1
```
The best model for the perceptron is
Perceptron(alpha=0.0001, class_weight=None, eta0=1.0, fit_intercept=True,
      max_iter=5000, n_iter=None, n_jobs=1, penalty='l1', random_state=0,
      shuffle=True, tol=None, verbose=0, warm_start=False)

It has the best cross-validation error/score, which is
0.9046195652173913

The entire cv_results:
{'mean_fit_time': array([5.24019928, 5.20779347, 5.08035491, 4.59420104, 4.98554964,
       5.13156815]), 'std_fit_time': array([0.02127728, 0.0436208 , 0.02531748, 0.04052317, 0.10893654,
       0.03874963]), 'mean_score_time': array([0.00088949, 0.00049522, 0.00035133, 0.00029767, 0.0004513 ,
       0.00039492]), 'std_score_time': array([0.00053087, 0.00044505, 0.00045175, 0.00045478, 0.00047294,
       0.00048377]), 
       
'param_alpha': masked_array(data=[0, 0.0001, 0.001, 0.01, 0.1, 1],
             mask=[False, False, False, False, False, False],
       fill_value='?',
            dtype=object), 
            
'params': [{'alpha': 0}, {'alpha': 0.0001}, {'alpha': 0.001}, {'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}], 
'split0_test_score': array([0.71544715, 0.91327913, 0.89159892, 0.62872629, 0.59620596,
       0.60704607]), 'split1_test_score': array([0.91847826, 0.9076087 , 0.91032609, 0.90217391, 0.77717391,
       0.74456522]), 'split2_test_score': array([0.7826087 , 0.91032609, 0.82065217, 0.86141304, 0.76630435,
       0.72826087]), 'split3_test_score': array([0.81521739, 0.91847826, 0.93206522, 0.89402174, 0.75543478,
       0.71195652]), 'split4_test_score': array([0.80978261, 0.91032609, 0.91576087, 0.66847826, 0.76902174,
       0.75      ]), 'split5_test_score': array([0.89130435, 0.89402174, 0.90217391, 0.6548913 , 0.77717391,
       0.74456522]), 'split6_test_score': array([0.89945652, 0.90217391, 0.88586957, 0.87228261, 0.76902174,
       0.64945652]), 'split7_test_score': array([0.88043478, 0.91304348, 0.91032609, 0.87228261, 0.73097826,
       0.48369565]), 'split8_test_score': array([0.83423913, 0.87228261, 0.87771739, 0.69565217, 0.70380435,
       0.46467391]), 'split9_test_score': array([0.89918256, 0.90463215, 0.86920981, 0.88010899, 0.52316076,
       0.59128065]), 
    'mean_test_score': array([0.84456522, 0.90461957, 0.89157609, 0.79293478, 0.71684783,
       0.64755435]), 'std_test_score': array([0.06136404, 0.0125549 , 0.02965372, 0.10865492, 0.0830868 ,
       0.10250881]), 'rank_test_score': array([3, 1, 2, 4, 5, 6]), 
       
    'split0_train_score': array([0.72606463, 0.89852008, 0.88402295, 0.64180006, 0.6013289 ,
       0.60857747]), 'split1_train_score': array([0.91394928, 0.89190821, 0.89160628, 0.88586957, 0.76449275,
       0.72977053]), 'split2_train_score': array([0.80344203, 0.92542271, 0.83454106, 0.88103865, 0.76811594,
       0.73158213]), 'split3_train_score': array([0.7928744 , 0.91938406, 0.92753623, 0.88586957, 0.7798913 ,
       0.74154589]), 'split4_train_score': array([0.79408213, 0.8946256 , 0.90911836, 0.66213768, 0.76177536,
       0.727657  ]), 'split5_train_score': array([0.9178744 , 0.89915459, 0.90911836, 0.69504831, 0.76841787,
       0.7294686 ]), 'split6_train_score': array([0.90398551, 0.92059179, 0.9057971 , 0.88496377, 0.7490942 ,
       0.62469807]), 'split7_train_score': array([0.88888889, 0.91878019, 0.90972222, 0.88194444, 0.77113527,
       0.48460145]), 'split8_train_score': array([0.86805556, 0.89281401, 0.8964372 , 0.75634058, 0.70138889,
       0.48671498]), 'split9_train_score': array([0.91608814, 0.91759734, 0.86598249, 0.89797766, 0.5249019 ,
       0.59160881]), 'mean_train_score': array([0.85253049, 0.90787986, 0.89338823, 0.80729903, 0.71905424,
       0.64562249]), 'std_train_score': array([0.06455277, 0.01279227, 0.02529028, 0.10063263, 0.08241117,
       0.09661164])}

After training on the whole training data, the training error is
0.8940217391304348

And the test error is
0.8805646036916395
```

## Perceptron 2
Using these params:
`{'alpha': [0, 0.00001, 0.0001]}`
```
The best model for the perceptron is
Perceptron(alpha=0.0001, class_weight=None, eta0=1.0, fit_intercept=True,
      max_iter=5000, n_iter=None, n_jobs=1, penalty='l1', random_state=0,
      shuffle=True, tol=None, verbose=0, warm_start=False)
It has the best cross-validation error/score, which is
0.9046195652173913
The entire cv_results:
{'mean_fit_time': array([5.3993021 , 7.39211228, 7.42554178]), 'std_fit_time': array([0.03588273, 1.31942264, 1.19874396]), 'mean_score_time': array([0.00079653, 0.00070086, 0.00050585]), 'std_score_time': array([0.00055704, 0.00038458, 0.0004318 ]), 'param_alpha': masked_array(data=[0, 1e-05, 0.0001],
             mask=[False, False, False],
       fill_value='?',
            dtype=object), 
            
'params': [{'alpha': 0}, {'alpha': 1e-05}, {'alpha': 0.0001}], 

'split0_test_score': array([0.71544715, 0.69105691, 0.91327913]), 'split1_test_score': array([0.91847826, 0.91576087, 0.9076087 ]), 'split2_test_score': array([0.7826087 , 0.89402174, 0.91032609]), 'split3_test_score': array([0.81521739, 0.9375    , 0.91847826]), 'split4_test_score': array([0.80978261, 0.92663043, 0.91032609]), 'split5_test_score': array([0.89130435, 0.92391304, 0.89402174]), 'split6_test_score': array([0.89945652, 0.91032609, 0.90217391]), 'split7_test_score': array([0.88043478, 0.9076087 , 0.91304348]), 'split8_test_score': array([0.83423913, 0.91032609, 0.87228261]), 'split9_test_score': array([0.89918256, 0.88555858, 0.90463215]), 

'mean_test_score': array([0.84456522, 0.89021739, 0.90461957]), 'std_test_score': array([0.06136404, 0.06802161, 0.0125549 ]), 'rank_test_score': array([3, 2, 1]), 'split0_train_score': array([0.72606463, 0.70552703, 0.89852008]), 'split1_train_score': array([0.91394928, 0.91817633, 0.89190821]), 'split2_train_score': array([0.80344203, 0.92330918, 0.92542271]), 'split3_train_score': array([0.7928744 , 0.92028986, 0.91938406]), 'split4_train_score': array([0.79408213, 0.91878019, 0.8946256 ]), 'split5_train_score': array([0.9178744 , 0.92421498, 0.89915459]), 'split6_train_score': array([0.90398551, 0.9160628 , 0.92059179]), 'split7_train_score': array([0.88888889, 0.91364734, 0.91878019]), 'split8_train_score': array([0.86805556, 0.92240338, 0.89281401]), 'split9_train_score': array([0.91608814, 0.91608814, 0.91759734]), 'mean_train_score': array([0.85253049, 0.89784992, 0.90787986]), 'std_train_score': array([0.06455277, 0.06418952, 0.01279227])}
After training on the whole training data, the training error is
0.8940217391304348
And the test error is
0.8805646036916395
```

# SVM
## Table for RBF SVM
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




## RBF
Using these params:
`{'gamma': [0.000001, 0.00001, 0.0001, 0.001], 'C': [10,100,1000], 'kernel': ['rbf']} `

```
The best model for SVM is
SVC(C=1000, cache_size=1000, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=1e-05, kernel='rbf',
  max_iter=-1, probability=False, random_state=0, shrinking=True,
  tol=0.001, verbose=False)

It has the best cross-validation error/score, which is
0.9206521739130434

The entire cv_results:
{'mean_fit_time': array([0.7041723 , 0.71049781, 0.64821832, 0.73430262, 0.74367635,
       0.6419502 , 0.61804199, 0.73285596, 0.72221398, 0.74133036,
       1.01790626, 1.22450743]), 'std_fit_time': array([0.00973669, 0.07660146, 0.04730662, 0.06278259, 0.03703049,
       0.03121752, 0.01703586, 0.01833368, 0.01662997, 0.03184142,
       0.05269744, 0.09420334]), 'mean_score_time': array([0.05717092, 0.05571861, 0.04809256, 0.0540935 , 0.05827482,
       0.04322937, 0.03375018, 0.03656375, 0.04016778, 0.0286402 ,
       0.02437127, 0.03390524]), 'std_score_time': array([0.00223258, 0.00386207, 0.00346428, 0.00989013, 0.01163095,
       0.00267033, 0.00163345, 0.00083887, 0.00061431, 0.00191514,
       0.0009544 , 0.00153305]), 
       
'param_C': masked_array(data=[10, 10, 10, 10, 100, 100, 100, 100, 1000, 1000, 1000,
                   1000],
             mask=[False, False, False, False, False, False, False, False,
                   False, False, False, False],
       fill_value='?',
            dtype=object), 
'param_gamma': masked_array(data=[1e-06, 1e-05, 0.0001, 0.001, 1e-06, 1e-05, 0.0001,
                   0.001, 1e-06, 1e-05, 0.0001, 0.001],
             mask=[False, False, False, False, False, False, False, False,
                   False, False, False, False],
       fill_value='?',
            dtype=object), 
'param_kernel': masked_array(data=['rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf',
                   'rbf', 'rbf', 'rbf', 'rbf'],
             mask=[False, False, False, False, False, False, False, False,
                   False, False, False, False],
       fill_value='?',
            dtype=object), 

'params': [{'C': 10, 'gamma': 1e-06, 'kernel': 'rbf'}, {'C': 10, 'gamma': 1e-05, 'kernel': 'rbf'}, {'C': 10, 'gamma': 0.0001, 'kernel': 'rbf'}, {'C': 10, 'gamma': 0.001, 'kernel': 'rbf'}, {'C': 100, 'gamma': 1e-06, 'kernel': 'rbf'}, {'C': 100, 'gamma': 1e-05, 'kernel': 'rbf'}, {'C': 100, 'gamma': 0.0001, 'kernel': 'rbf'}, {'C': 100, 'gamma': 0.001, 'kernel': 'rbf'}, {'C': 1000, 'gamma': 1e-06, 'kernel': 'rbf'}, {'C': 1000, 'gamma': 1e-05, 'kernel': 'rbf'}, {'C': 1000, 'gamma': 0.0001, 'kernel': 'rbf'}, {'C': 1000, 'gamma': 0.001, 'kernel': 'rbf'}], 

'split0_test_score': array([0.71815718, 0.74525745, 0.87262873, 0.89701897, 0.77235772,
       0.87804878, 0.92140921, 0.90785908, 0.88888889, 0.92682927,
       0.93495935, 0.88888889]), 'split1_test_score': array([0.74184783, 0.78804348, 0.88043478, 0.89402174, 0.79891304,
       0.89402174, 0.91847826, 0.91304348, 0.91032609, 0.94021739,
       0.91304348, 0.90217391]), 'split2_test_score': array([0.72554348, 0.76902174, 0.86141304, 0.88858696, 0.79619565,
       0.88315217, 0.91847826, 0.89673913, 0.88858696, 0.91304348,
       0.91032609, 0.88315217]), 'split3_test_score': array([0.72826087, 0.73913043, 0.85869565, 0.88043478, 0.76086957,
       0.88043478, 0.91032609, 0.92119565, 0.89402174, 0.92934783,
       0.91032609, 0.91304348]), 'split4_test_score': array([0.74456522, 0.77717391, 0.85054348, 0.88315217, 0.80163043,
       0.87228261, 0.9076087 , 0.91032609, 0.88043478, 0.92119565,
       0.92119565, 0.91847826]), 'split5_test_score': array([0.74456522, 0.77717391, 0.85869565, 0.85597826, 0.78532609,
       0.86956522, 0.88315217, 0.87771739, 0.87771739, 0.92119565,
       0.91576087, 0.88043478]), 'split6_test_score': array([0.76086957, 0.77717391, 0.84782609, 0.86141304, 0.79347826,
       0.87228261, 0.89673913, 0.89402174, 0.89130435, 0.88858696,
       0.91032609, 0.89130435]), 'split7_test_score': array([0.70380435, 0.72282609, 0.86956522, 0.86956522, 0.76630435,
       0.89673913, 0.91847826, 0.9048913 , 0.90217391, 0.94021739,
       0.93478261, 0.91576087]), 'split8_test_score': array([0.74184783, 0.76086957, 0.86413043, 0.875     , 0.79076087,
       0.89130435, 0.91304348, 0.89673913, 0.9048913 , 0.92663043,
       0.92119565, 0.875     ]), 'split9_test_score': array([0.73297003, 0.76566757, 0.82833787, 0.86376022, 0.77656676,
       0.85831063, 0.89645777, 0.88555858, 0.86648501, 0.89918256,
       0.91825613, 0.89100817]), 
'params': [{'C': 10, 'gamma': 1e-06, 'kernel': 'rbf'}, {'C': 10, 'gamma': 1e-05, 'kernel': 'rbf'}, {'C': 10, 'gamma': 0.0001, 'kernel': 'rbf'}, {'C': 10, 'gamma': 0.001, 'kernel': 'rbf'}, {'C': 100, 'gamma': 1e-06, 'kernel': 'rbf'}, {'C': 100, 'gamma': 1e-05, 'kernel': 'rbf'}, {'C': 100, 'gamma': 0.0001, 'kernel': 'rbf'}, {'C': 100, 'gamma': 0.001, 'kernel': 'rbf'}, {'C': 1000, 'gamma': 1e-06, 'kernel': 'rbf'}, {'C': 1000, 'gamma': 1e-05, 'kernel': 'rbf'}, {'C': 1000, 'gamma': 0.0001, 'kernel': 'rbf'}, {'C': 1000, 'gamma': 0.001, 'kernel': 'rbf'}], 
'mean_test_score': array([0.73423913, 0.76222826, 0.85923913, 0.87690217, 0.78423913,
       0.87961957, 0.90842391, 0.90081522, 0.89048913, 0.92065217,
       0.91902174, 0.89592391]), 
       'std_test_score': array([0.01528656, 0.0194195 , 0.01386744, 0.01339535, 0.01363035,
       0.01146404, 0.01189943, 0.01251694, 0.01263183, 0.01568937,
       0.00886767, 0.01473177]), 'rank_test_score': array([12, 11,  9,  8, 10,  7,  3,  4,  6,  1,  2,  5]), 
       
'split0_train_score': array([0.73754153, 0.77136817, 0.87043189, 0.93385684, 0.78828149,
       0.88764724, 0.93476291, 0.97765026, 0.8961039 , 0.93627303,
       0.96103896, 0.99063727]), 'split1_train_score': array([0.73792271, 0.76781401, 0.8740942 , 0.93115942, 0.78925121,
       0.89281401, 0.93387681, 0.97584541, 0.89221014, 0.93870773,
       0.95923913, 0.99094203]), 'split2_train_score': array([0.73762077, 0.7705314 , 0.86865942, 0.93327295, 0.78653382,
       0.89009662, 0.93478261, 0.97675121, 0.8955314 , 0.9365942 ,
       0.96135266, 0.99094203]), 'split3_train_score': array([0.73943237, 0.76992754, 0.87379227, 0.93417874, 0.78804348,
       0.88768116, 0.93387681, 0.97644928, 0.8955314 , 0.93176329,
       0.96074879, 0.99154589]), 'split4_train_score': array([0.73580918, 0.77355072, 0.87560386, 0.93448068, 0.78804348,
       0.89221014, 0.93297101, 0.97433575, 0.89402174, 0.93357488,
       0.9580314 , 0.98943237]), 'split5_train_score': array([0.73580918, 0.77536232, 0.88043478, 0.93599034, 0.790157  ,
       0.89221014, 0.93327295, 0.97735507, 0.89311594, 0.93780193,
       0.96044686, 0.9906401 ]), 'split6_train_score': array([0.73731884, 0.7705314 , 0.87741546, 0.93478261, 0.78743961,
       0.8946256 , 0.93417874, 0.97795894, 0.8955314 , 0.93568841,
       0.96105072, 0.99154589]), 'split7_train_score': array([0.74214976, 0.76992754, 0.86775362, 0.93387681, 0.78562802,
       0.88375604, 0.93085749, 0.97493961, 0.89221014, 0.93478261,
       0.9580314 , 0.98913043]), 'split8_train_score': array([0.73641304, 0.7705314 , 0.8759058 , 0.93538647, 0.79045894,
       0.89039855, 0.93478261, 0.97795894, 0.89190821, 0.93629227,
       0.96105072, 0.99245169]), 'split9_train_score': array([0.7364926 , 0.76878962, 0.87594325, 0.93450045, 0.78871114,
       0.89224268, 0.93359493, 0.97675823, 0.89586478, 0.93359493,
       0.961968  , 0.99185029]), 
       
'mean_train_score': array([0.737651  , 0.77083341, 0.87400346, 0.93414853, 0.78825482,
       0.89036822, 0.93369569, 0.97660027, 0.89420291, 0.93550733,
       0.96029587, 0.9909118 ]), 'std_train_score': array([0.00182396, 0.00208301, 0.00378649, 0.00124186, 0.00142457,
       0.00304673, 0.00112116, 0.00117857, 0.00161531, 0.00199647,
       0.00131167, 0.00097822])}

After training on the whole training data, the training error is
0.9345108695652173
And the test error is
0.9011943539630836
```

# Raw data
## SVM polynomial degree 2
```
SVC deg2 w C:1,gamma:1e-05: 0.690215 (0.016427)
SVC deg2 w C:1,gamma:0.0001: 0.801080 (0.014320)
SVC deg2 w C:1,gamma:0.001: 0.909773 (0.012914)
```

## SVM rbf
```
SVC rbf:10000,gamma:1e-06: 0.926083 (0.011606)
SVC rbf:10000,gamma:1e-05: 0.931519 (0.010314)
SVC rbf:10000,gamma:0.0001: 0.917659 (0.011167)
SVC rbf:10000,gamma:0.001: 0.886146 (0.018858)
SVC rbf:100000,gamma:1e-06: 0.934780 (0.008613)
SVC rbf:100000,gamma:1e-05: 0.933695 (0.008354)
SVC rbf:100000,gamma:0.0001: 0.912769 (0.009154)
SVC rbf:100000,gamma:0.001: 0.875008 (0.016825)
SVC rbf:1,gamma:1e-06: 0.705972 (0.019797)
SVC rbf:1,gamma:1e-05: 0.727993 (0.019549)
SVC rbf:1,gamma:0.0001: 0.757069 (0.018867)
SVC rbf:1,gamma:0.001: 0.809503 (0.009675)
Best model is:
SVC(C=100000, cache_size=2000, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=1e-06, kernel='rbf',
  max_iter=-1, probability=False, random_state=0, shrinking=True,
  tol=0.001, verbose=False)
With a test score of: 0.9196525515743756
Training score: 0.9451086956521739
```
SVM with more C's:
```
SVC rbf:1000000,gamma:1e-06: 0.938855 (0.008018)
SVC rbf:1000000,gamma:1e-05: 0.924722 (0.009305)
SVC rbf:1000000,gamma:0.0001: 0.908963 (0.015209)
SVC rbf:1000000,gamma:0.001: 0.866043 (0.018327)
SVC rbf:10000000,gamma:1e-06: 0.932602 (0.011615)
SVC rbf:10000000,gamma:1e-05: 0.914394 (0.010827)
SVC rbf:10000000,gamma:0.0001: 0.893753 (0.019105)
SVC rbf:10000000,gamma:0.001: 0.860611 (0.020472)
Best model is:
SVC(**C=1000 000**, cache_size=2000, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, **gamma=1e-06**, kernel='rbf',
  max_iter=-1, probability=False, random_state=0, shrinking=True,
  tol=0.001, verbose=False)
With a test score of: 0.9163952225841476
Training score: **0.9548913043478261**
```

SVM with more gammas
```
SVC rbf:10000000,gamma:1e-08: 0.914397 (0.011660)
SVC rbf:10000000,gamma:1e-07: 0.928257 (0.012187)
SVC rbf:1000000,gamma:1e-08: 0.917119 (0.008002)
SVC rbf:1000000,gamma:1e-07: 0.932058 (0.011763)
SVC rbf:100000,gamma:1e-08: 0.883692 (0.012605)
SVC rbf:100000,gamma:1e-07: 0.922279 (0.011634)
SVC rbf:10000,gamma:1e-08: 0.775004 (0.013612)
SVC rbf:10000,gamma:1e-07: 0.886137 (0.012237)
SVC rbf:1000,gamma:1e-08: 0.735058 (0.014414)
SVC rbf:1000,gamma:1e-07: 0.785334 (0.015061)
SVC rbf:100,gamma:1e-08: 0.711684 (0.015023)
SVC rbf:100,gamma:1e-07: 0.732884 (0.014195)
SVC rbf:10,gamma:1e-08: 0.679075 (0.018185)
SVC rbf:10,gamma:1e-07: 0.719831 (0.021452)
SVC rbf:1,gamma:1e-08: 0.658423 (0.014710)
SVC rbf:1,gamma:1e-07: 0.680428 (0.019907)
Best model is:
SVC(C=1000000, cache_size=2000, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=1e-07, kernel='rbf',
  max_iter=-1, probability=False, random_state=0, shrinking=True,
  tol=0.001, verbose=False)
With a test score of: 0.9196525515743756
Training score: 0.9404891304347827
```

## SVM RBF scaled
```
SVC rbf C:10000000,gamma:1e-08: 0.907875 (0.014746)
SVC rbf C:10000000,gamma:1e-07: 0.918201 (0.009414)
SVC rbf C:10000000,gamma:1e-06: 0.916299 (0.011239)
SVC rbf C:10000000,gamma:1e-05: 0.931786 (0.011080)
SVC rbf C:10000000,gamma:0.0001: 0.921196 (0.012861)
SVC rbf C:1000000,gamma:1e-08: 0.914939 (0.015895)
SVC rbf C:1000000,gamma:1e-07: 0.913580 (0.012501)
SVC rbf C:1000000,gamma:1e-06: 0.921465 (0.006737)
SVC rbf C:1000000,gamma:1e-05: 0.935055 (0.008366)
SVC rbf C:1000000,gamma:0.0001: 0.931519 (0.015610)
SVC rbf C:100000,gamma:1e-08: 0.899722 (0.013822)
SVC rbf C:100000,gamma:1e-07: 0.922594 (0.011712)
SVC rbf C:100000,gamma:1e-06: 0.927988 (0.010549)
SVC rbf C:100000,gamma:1e-05: 0.934235 (0.009663)
SVC rbf C:100000,gamma:0.0001: 0.937498 (0.011603)
SVC rbf C:10000,gamma:1e-08: 0.811679 (0.017298)
SVC rbf C:10000,gamma:1e-07: 0.905430 (0.013174)
SVC rbf C:10000,gamma:1e-06: 0.922277 (0.012387)
SVC rbf C:10000,gamma:1e-05: 0.932606 (0.010387)
SVC rbf C:10000,gamma:0.0001: 0.934510 (0.007926)
SVC rbf C:1000,gamma:1e-08: 0.608424 (0.000559)
SVC rbf C:1000,gamma:1e-07: 0.814669 (0.017701)
SVC rbf C:1000,gamma:1e-06: 0.906246 (0.012794)
SVC rbf C:1000,gamma:1e-05: 0.924179 (0.012585)
SVC rbf C:1000,gamma:0.0001: 0.932877 (0.011434)
Best model is:
SVC(C=100000, cache_size=2000, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.0001, kernel='rbf',
  max_iter=-1, probability=False, random_state=0, shrinking=True,
  tol=0.001, verbose=False)
With a test score of: 0.9381107491856677
Training score: 0.9611413043478261
```

MOARRR
```
SVC rbf C:10000000,gamma:0.001: 0.903256 (0.011962)
SVC rbf C:10000000,gamma:0.01: 0.889673 (0.021819)
SVC rbf C:1000000,gamma:0.001: 0.916844 (0.013895)
SVC rbf C:1000000,gamma:0.01: 0.898095 (0.017803)
SVC rbf C:100000,gamma:0.001: 0.924723 (0.012665)
SVC rbf C:100000,gamma:0.01: 0.898905 (0.023188)
SVC rbf C:10000,gamma:0.001: 0.931517 (0.015384)
SVC rbf C:10000,gamma:0.01: 0.915482 (0.016683)
SVC rbf C:1000,gamma:0.001: 0.936954 (0.012216)
SVC rbf C:1000,gamma:0.01: 0.926079 (0.015687)
Best model is:
SVC(C=1000, cache_size=2000, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',
  max_iter=-1, probability=False, random_state=0, shrinking=True,
  tol=0.001, verbose=False)
With a test score of: 0.9359391965255157
Training score: 0.9619565217391305
```

## Perceptron scaled
```
The best model for the perceptron is
Perceptron(alpha=0, class_weight=None, eta0=1.0, fit_intercept=True,
      max_iter=5000, n_iter=None, n_jobs=1, penalty='l1', random_state=0,
      shuffle=True, tol=None, verbose=0, warm_start=False)
It has the best cross-validation error/score, which is
0.8986413043478261
The entire cv_results:
{'mean_fit_time': array([5.83175068, 5.4704551 , 6.00584443, 7.33051083, 7.2788177 ,
       7.91346514, 8.46173687]), 'std_fit_time': array([0.23843888, 0.11458415, 0.70055534, 0.39029501, 0.28323448,
       0.29479253, 0.48933741]), 'mean_score_time': array([0.00110271, 0.00029492, 0.0004005 , 0.0005353 , 0.00070844,
       0.00055513, 0.00060847]), 'std_score_time': array([0.00202711, 0.00045059, 0.00037547, 0.0001567 , 0.00033339,
       0.00015085, 0.00045262]), 'param_alpha': masked_array(data=[0, 1e-05, 0.0001, 0.001, 0.01, 0.1, 1],
             mask=[False, False, False, False, False, False, False],
       fill_value='?',
            dtype=object), 'params': [{'alpha': 0}, {'alpha': 1e-05}, {'alpha': 0.0001}, {'alpha': 0.001}, {'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}], 'split0_test_score': array([0.89701897, 0.90514905, 0.91869919, 0.89159892, 0.84552846,
       0.78319783, 0.60704607]), 'split1_test_score': array([0.91576087, 0.91032609, 0.9048913 , 0.8125    , 0.875     ,
       0.60869565, 0.60869565]), 'split2_test_score': array([0.89130435, 0.875     , 0.89130435, 0.87771739, 0.83423913,
       0.82608696, 0.60869565]), 'split3_test_score': array([0.91576087, 0.91847826, 0.93206522, 0.92391304, 0.92119565,
       0.61141304, 0.60869565]), 'split4_test_score': array([0.92391304, 0.89130435, 0.9048913 , 0.89130435, 0.81521739,
       0.79619565, 0.60869565]), 'split5_test_score': array([0.92663043, 0.89130435, 0.88043478, 0.86413043, 0.82608696,
       0.60869565, 0.60869565]), 'split6_test_score': array([0.88858696, 0.83695652, 0.86956522, 0.8451087 , 0.79347826,
       0.60869565, 0.60869565]), 'split7_test_score': array([0.88586957, 0.87771739, 0.875     , 0.86413043, 0.86141304,
       0.60869565, 0.39130435]), 'split8_test_score': array([0.89945652, 0.9076087 , 0.89130435, 0.85326087, 0.82336957,
       0.70652174, 0.39130435]), 'split9_test_score': array([0.84196185, 0.89100817, 0.88010899, 0.8773842 , 0.8746594 ,
       0.71934605, 0.60762943]), 'mean_test_score': array([0.8986413 , 0.89048913, 0.89483696, 0.8701087 , 0.84701087,
       0.68777174, 0.56494565]), 'std_test_score': array([0.02352315, 0.02226172, 0.01907865, 0.02866534, 0.03506447,
       0.08497428, 0.08682234]), 'rank_test_score': array([1, 3, 2, 4, 5, 6, 7]), 'split0_train_score': array([0.90456056, 0.88674117, 0.9063727 , 0.90154032, 0.86439142,
       0.77620054, 0.60857747]), 'split1_train_score': array([0.91032609, 0.90942029, 0.90217391, 0.83333333, 0.87771739,
       0.60839372, 0.60839372]), 'split2_train_score': array([0.92179952, 0.92783816, 0.92330918, 0.91878019, 0.83967391,
       0.83967391, 0.60839372]), 'split3_train_score': array([0.91908213, 0.90398551, 0.91727053, 0.90428744, 0.88707729,
       0.61020531, 0.60839372]), 'split4_train_score': array([0.91938406, 0.90126812, 0.90217391, 0.88677536, 0.8031401 ,
       0.78743961, 0.60839372]), 'split5_train_score': array([0.91455314, 0.88164251, 0.89100242, 0.88073671, 0.79891304,
       0.60839372, 0.60839372]), 'split6_train_score': array([0.90881643, 0.85295894, 0.87952899, 0.84752415, 0.80495169,
       0.60839372, 0.60839372]), 'split7_train_score': array([0.87922705, 0.87439614, 0.88254831, 0.86322464, 0.84601449,
       0.60839372, 0.39160628]), 'split8_train_score': array([0.88737923, 0.91062802, 0.8964372 , 0.85597826, 0.78834541,
       0.7201087 , 0.39160628]), 'split9_train_score': array([0.89133716, 0.91367341, 0.90582554, 0.90552369, 0.87232116,
       0.74916994, 0.60851192]), 'mean_train_score': array([0.90564653, 0.89625523, 0.90066427, 0.87977041, 0.83825459,
       0.69163729, 0.56506643]), 'std_train_score': array([0.01406787, 0.02102878, 0.01318503, 0.02710362, 0.03496807,
       0.08758632, 0.08673009])}
After training on the whole training data, the training error is
0.9133152173913044
And the test error is
0.8979370249728555
```