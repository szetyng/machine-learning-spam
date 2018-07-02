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