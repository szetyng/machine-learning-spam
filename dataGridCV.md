Perceptron:

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