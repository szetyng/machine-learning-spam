# logbook

- [logbook](#logbook)
    - [Todo](#todo)
    - [Notes about SK-learn](#notes-about-sk-learn)
        - [Model selection](#model-selection)
            - [cross-validate](#cross-validate)
            - [cross_val_score](#crossvalscore)
            - [GridSearchCV](#gridsearchcv)
        - [SVM](#svm)
            - [SVC](#svc)
        - [Linear_model](#linearmodel)
            - [Perceptron](#perceptron)
            - [SGDClassifier](#sgdclassifier)
        - [On scaling](#on-scaling)
    - [Project email spam](#project-email-spam)
        - [Error measure](#error-measure)
    - [Project movie](#project-movie)
        - [Input](#input)
        - [Output](#output)
    - [Data collection](#data-collection)
        - [Perceptron:](#perceptron)
        - [SVM](#svm)
    - [To add or not to add](#to-add-or-not-to-add)
        - [Problem definition](#problem-definition)
        - [Validation error](#validation-error)
        - [Perceptron stopping criterion](#perceptron-stopping-criterion)

## Todo
1. Plot learning curves (test and training errors vs number of samples). Hopefully it will help in explaining overfitting.
2. Collect all data (cv_errors for each parameter in both perceptron and nn, training error of the perceptron and nn using the paramater chosen, test error to evaluate overall performance on unseen data)
3. Try plotting cross-validation error for regularisation parameters like lambda?
4. Write equations for confidence interval if possible
5. Finish report!

## Notes about SK-learn
### [Model selection](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection)
#### [cross-validate](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate)
`cross_validate(estimator, X, y=None, groups=None, scoring=None, cv=None, n_jobs=1, verbose=0, fit_params=None, pre_dispatch=‘2*n_jobs’, return_train_score=’warn’)`  
User guide [here](http://scikit-learn.org/stable/modules/cross_validation.html)

Parameters:
- `X`: from training set, x
- `y`: from training set, y
- `cv`: k-fold number

Returns:
- `scores`: a dictionary with the following keys. Each key accesses an array of scores. The number of scores in the array is the number of validation sets you set in CV. To get cross-validation error, take the average of the `test_score`, which is the validation error for each validation set. Training score is the training error for training the hypothesis on the smaller training set, validation error is the 'test error' when evaluating that hypothesis' performance on the small held out 'test set' / validation set.  
`test_score`  
`train_score`  
`fit_time`  
`score_time`  

The whole thing about how this is different from `cross_val_score` is something about this being able to do CV on different metrics. Metrics seems to be about the 'scoring' parameter, which is not what I'm aiming to do. I would like to run CV on different parameters to the estimator. 

#### [cross_val_score](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html#sklearn.model_selection.cross_val_score)
`cross_val_score(estimator, X, y=None, groups=None, scoring=None, cv=None, n_jobs=1, verbose=0, fit_params=None, pre_dispatch=‘2*n_jobs’)`  

User guide [here](http://scikit-learn.org/stable/modules/cross_validation.html#cross-validation)

  
Parameters:
- `X`: from training set, x
- `y`: from training set, y
- `cv`: k-fold number

Returns:
- `scores`: an array of scores. The number of scores in this array is the number of CV-folds we set. The score is the validation error of each CV-fold, to get the cross-validation error, we take the average of this whole array. Each score is equivalent to `test_score` in `cross_validate` ie they are validation errors, so the method of finding the cross-validation error is the same.

#### [GridSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV)
`GridSearchCV(estimator, param_grid, scoring=None, fit_params=None, n_jobs=1, iid=True, refit=True, cv=None, verbose=0, pre_dispatch=‘2*n_jobs’, error_score=’raise’, return_train_score=’warn’)`  
User guide [here](http://scikit-learn.org/stable/modules/grid_search.html#grid-search)   
  
Implements a 'fit' and a 'score' method. Parameters of the estimator used to apply these methods are optimised by cv grid-search over a parameter grid.  

Parameters:
- `param_grid`: dictionary / list of dictionaries.   
Keys: Parameters names (string)  
Values: lists of parameter settings to try  
List of dictionaries means trying more than one type of parameters.
- `cv`
- `refit`: boolean, default=True.
- `return_train_score`: default=warn. Training scores are the training errors on the smaller training set, test scores are the validation errors on the held out validation set. Not needed?

### [SVM](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.svm)
#### [SVC](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
`SVC(C=1.0, kernel=’rbf’, degree=3, gamma=’auto’, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=’ovr’, random_state=None)`  

User guide [here](http://scikit-learn.org/stable/modules/svm.html#svm-classification)

Important parameters for RBF kernel:  
Important ones to tune are `C` and `gamma`. Notes about kernel functions [here](http://scikit-learn.org/stable/modules/svm.html#kernel-functions). There is also a note about RBF parameters in that link. Proper choices of `C` and `gamma` are [critical to RBF performance](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV). More thorough explanation about them [here](http://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html#sphx-glr-auto-examples-svm-plot-rbf-parameters-py)
- `C`: float. Penalty parameter of error term. If there's a lot of noise, decrease it - corresponds to more regularisation.   
**High C**: less regularisation. Wants to decrease training error, harder margin SVM. Trading smaller margins to get better classification. More support vectors.   
**Low C**: more regularisation. Softer margin SVM. Trading worse training error for larger margins. Less support vectors. 
- `gamma`: float. Kernel coefficient for 'rbf', 'poly'. If gamma is auto, then 1/n_features will be used.   
**_For RBF_**: Gamma must be larger than 0. Defines how much influence a single training point has.   
**If gamma large**, the less effect the point has, the other examples have to be closer to it in order to be affected. As a result, there will be more support vectors (because the SV influence is small) and will be in danger of overfitting.  
**If gamma small**, any selected SV would have a large effect, could include the whole data set. 
- Example: gamma small, C large. Gamma small means smoother model, can be made more complex by making C large ie less regularisation, which increase nr of support vectors. 

Other parameters:
Important ones to tune are `C` and `gamma`, and `coef0` if using poly
- `kernel`: string. Some of the options are: 'linear', 'poly', 'rbf'
- `degree`: Degree of the polynomial kernel function. Ignored if some other kernel is used
- `coef0`: Independent term in kernel function when 'poly' is used
- `shrinking`: default: true. Whether to shrink the heuristic. 
- `tol`: tolerance for stopping criterion. default: 0.001
- `random_state`
- `cache_size`: in MB. Can change to 500 or 1000 instead of the default 200, if you have enough RAM available. Helps in speed.

Attributes
- `support_`: Array with the indices of the support vectors. Height of array = nr of support vectors.
- `support_vectors_`: Array with the support vectors. Height of array = nr of support vectors. Length of array = nr of features. One row: one support vector with all its features.
- `n_support_`: Array with the nr of support vectors for each class. Height of array = nr of classes. 
- `dual_coef_`
- `coef_`: only available in linear kernel
- `intercept_`

- Look into scaling/normalising your data because SVC is not scale-invariant. Guide to pre-procession data [here](http://scikit-learn.org/stable/modules/preprocessing.html#preprocessing). Same scaling must be applied to test vector


### Linear_model
#### [Perceptron](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html)
`Perceptron(penalty=None, alpha=0.0001, fit_intercept=True, max_iter=None, tol=None, shuffle=True, verbose=0, eta0=1.0, n_jobs=1, random_state=0, class_weight=None, warm_start=False, n_iter=None)`  
User guide [here](http://scikit-learn.org/stable/modules/linear_model.html#perceptron)

Updates its model only on mistakes. If linearly separable, it will stop before max_iterations. If not linearly separable, then it is important to stop it or else it will go on forever.  

Important parameters to tune by CV:  
No penalty, know that L1 is better because of sparse dataset.
- `alpha`: Constant that multiplies the regularisation term, if used.

Other parameters: 
- `penalty`: 'l2', 'l1', 'elasticnet' or None. 
- `fit_intercept`: Default: True. Assumes that data is not centered. 
- `max_iter`: Max number of passes over the training data aka nr of epochs. Defaults to 1000 if `tol` is not None, if `tol` is none, must specify `max_iter`
- `tol`: Defaults to 0.001. Iterations will stop when loss > prev loss - tol
- `random_state`

Perceptron and `SGDClassifier` share the same underlying principle if `SGDClassifier(loss=”perceptron”, eta0=1, learning_rate=”constant”, penalty=None)`.


#### [SGDClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier)
`SGDClassifier(loss=’hinge’, penalty=’l2’, alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=None, tol=None, shuffle=True, verbose=0, epsilon=0.1, n_jobs=1, random_state=None, learning_rate=’optimal’, eta0=0.0, power_t=0.5, class_weight=None, warm_start=False, average=False, n_iter=None)`  
User guide [here](http://scikit-learn.org/stable/modules/sgd.html#sgd)

### On scaling
```
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)  # Don't cheat - fit only on training data
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)  # apply same transformation to test data
```

## Project email spam
### Error measure
Loss function that reflects the potential use of the predictor. Training error and test error, too. 

If linear regression, binary error. If logistic regression (i.e. probability of data belonging to a discrete class), then, cross entropy error.

Scikit perceptron info about optimisation:   
1. https://stats.stackexchange.com/questions/143996/what-is-the-difference-between-linear-perceptron-regression-and-ls-linear-regres 
2. https://www.quora.com/What-is-the-loss-function-of-the-standard-perceptron-algorithm

Loss function for perceptron:  
1. [accuracy_score](http://scikit-learn.org/stable/modules/model_evaluation.html#accuracy-score)
2. [look at model validation](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection)
3. [plot test and training errors](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.learning_curve.html#sklearn.model_selection.learning_curve)






## Project movie
### Input
`movie-genre` shows that there are 18 genres, i.e. the vector x has 18 features (`d = 18`)

### Output
`ratings-test` and `ratings-train` have three columns each:  
|User ID    |Movie ID   |Rating  |   
|-----------|-----------|--------|

Test data has 30,002 points.  
Training data has 70,002 points.  
The total data set was split into 70% training and 30% testing. 

There are 9066 movies, and 671 users.

## Data collection
### Perceptron:
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
            dtype=object), 'params': [{'alpha': 0}, {'alpha': 1e-05}, {'alpha': 0.0001}], 'split0_test_score': array([0.71544715, 0.69105691, 0.91327913]), 'split1_test_score': array([0.91847826, 0.91576087, 0.9076087 ]), 'split2_test_score': array([0.7826087 , 0.89402174, 0.91032609]), 'split3_test_score': array([0.81521739, 0.9375    , 0.91847826]), 'split4_test_score': array([0.80978261, 0.92663043, 0.91032609]), 'split5_test_score': array([0.89130435, 0.92391304, 0.89402174]), 'split6_test_score': array([0.89945652, 0.91032609, 0.90217391]), 'split7_test_score': array([0.88043478, 0.9076087 , 0.91304348]), 'split8_test_score': array([0.83423913, 0.91032609, 0.87228261]), 'split9_test_score': array([0.89918256, 0.88555858, 0.90463215]), 'mean_test_score': array([0.84456522, 0.89021739, 0.90461957]), 'std_test_score': array([0.06136404, 0.06802161, 0.0125549 ]), 'rank_test_score': array([3, 2, 1]), 'split0_train_score': array([0.72606463, 0.70552703, 0.89852008]), 'split1_train_score': array([0.91394928, 0.91817633, 0.89190821]), 'split2_train_score': array([0.80344203, 0.92330918, 0.92542271]), 'split3_train_score': array([0.7928744 , 0.92028986, 0.91938406]), 'split4_train_score': array([0.79408213, 0.91878019, 0.8946256 ]), 'split5_train_score': array([0.9178744 , 0.92421498, 0.89915459]), 'split6_train_score': array([0.90398551, 0.9160628 , 0.92059179]), 'split7_train_score': array([0.88888889, 0.91364734, 0.91878019]), 'split8_train_score': array([0.86805556, 0.92240338, 0.89281401]), 'split9_train_score': array([0.91608814, 0.91608814, 0.91759734]), 'mean_train_score': array([0.85253049, 0.89784992, 0.90787986]), 'std_train_score': array([0.06455277, 0.06418952, 0.01279227])}
After training on the whole training data, the training error is
0.8940217391304348
And the test error is
0.8805646036916395
```
### SVM
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
       0.91825613, 0.89100817]), 'mean_test_score': array([0.73423913, 0.76222826, 0.85923913, 0.87690217, 0.78423913,
       0.87961957, 0.90842391, 0.90081522, 0.89048913, 0.92065217,
       0.91902174, 0.89592391]), 'std_test_score': array([0.01528656, 0.0194195 , 0.01386744, 0.01339535, 0.01363035,
       0.01146404, 0.01189943, 0.01251694, 0.01263183, 0.01568937,
       0.00886767, 0.01473177]), 'rank_test_score': array([12, 11,  9,  8, 10,  7,  3,  4,  6,  1,  2,  5]), 'split0_train_score': array([0.73754153, 0.77136817, 0.87043189, 0.93385684, 0.78828149,
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


## To add or not to add
### Problem definition
\subsection{Problem definition}
The task of predicting if new emails are spam or ham is a binary classification problem. Classification given in the dataset is 1 for spam and 0 for ham. In order for us to formulate the problem, we will convert the 0's into -1's. 

\noindent Input: $\bm x_1, \bm x_2, ... \bm x_n, where \  x = x_1, x_2, ... x_{57}$ \\
\noindent Output: $y_1, y_2, ... y_n, where \  y=\{-1, 1\}$ \\
\noindent Target function: $x \rightarrow y$  \\
\noindent Hypothesis: $h(x) = \hat{y} = sign(w^Tx) = ywtx thing$ \\

Thus, the appropriate loss function used to analyse the quality of the machine learning algorithm is the binary error:

\begin{equation}
\label{binaryerr}
l(\hat{y},y) = \mathbb{I}(\hat{y}\neq y)
\end{equation}

The goal of the problem is to maximise the probability of the event that the machine learning model correctly predicts new emails, or mathematically, to minimise the test error of the model. 

\begin{align}
\label{testerr}
R(w) &= \mathbb{E}l(\hat{y},y) \\
&= \mathbb{E}[\mathbb{I}(\hat{y}\neq y)
\end{align}

The test error is obtained when evaluating the model on unseen data, it is not made available during the training period. During training, empirical risk minimisation methods are employed in order to minimise the training error.

\begin{align}
\label{trainingerr}
\hat{R_n}(w) &= \frac{1}{n}\sum_{i=1}^nl(\hat{y},y) \\
&= \frac{1}{n}\sum_{i=1}^n\mathbb{I}(\hat{y}\neq y)
\end{align}

The relationship between the training and test errors can be seen by manipulating Hoeffding's inequality.

With probability at least 1 - $\delta$:
\begin{equation}
|\hat{R_n}(h)-R(h)| \leq \sqrt{\frac{\log\frac{2M}{\delta}}{2n}}
\end{equation}

Or is it, empirical risk minimisation, with probability at least 1 - $\delta$:
\begin{equation}
R(g) - R(h^*) \leq \sqrt{\frac{1\log\frac{2M}{\delta}}{n}}
\end{equation}

### Validation error
If the validation set is too small, the validation error would not be good at estimating the test error of the small training set. If the validation set is too large, even though the validation error estimates the test error of the small training set well, it does not track the actual test error.

### Perceptron stopping criterion
It is clear that the algorithm will stop after some time if the data is linearly separable. If the data is not linearly separable, the perceptron will never converge so it must be forced to stop after a certain criteria is met. This criterion could be :
\begin{itemize}
	\item A hard limit on the maximum number of epochs that the model is allowed to pass, or
    \item A certain tolerance for the decrease in training error when the weight vector is being updated. If the update results in a decrease in training error is that lower than the tolerance, the perceptron should stop attempting to decrease it further.
\end{itemize}