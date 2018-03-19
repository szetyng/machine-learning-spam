# logbook

## Todo
1. Plot learning curves (test and training errors vs number of samples). Hopefully it will help in explaining overfitting.
2. Collect all data (cv_errors for each parameter in both perceptron and nn, training error of the perceptron and nn using the paramater chosen, test error to evaluate overall performance on unseen data)
3. Try plotting cross-validation error for regularisation parameters like lambda?
4. Try implementing SVM again, I don't understand neural networks :(
4. Understand backpropagation, it is used in MLPClassifier!
5. Figure out how many hidden layers to use and WHY
5. Write equations for confidence interval if possible
6. Finish report!

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





