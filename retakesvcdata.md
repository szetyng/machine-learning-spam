```
SVC deg2 w C:1,gamma:1e-05: 0.690215 (0.016427)
SVC deg2 w C:1,gamma:0.0001: 0.801080 (0.014320)
SVC deg2 w C:1,gamma:0.001: 0.909773 (0.012914)
```

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
SVC(C=1000000, cache_size=2000, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=1e-06, kernel='rbf',
  max_iter=-1, probability=False, random_state=0, shrinking=True,
  tol=0.001, verbose=False)
With a test score of: 0.9163952225841476
Training score: 0.9548913043478261
```

SVM with more gammas"
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