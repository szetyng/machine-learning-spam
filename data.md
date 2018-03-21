For perceptron
```
alphas = [0, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
L2:
P0: 0.888859 (0.036467)
P1: 0.667391 (0.077321)
P2: 0.664402 (0.084189)
P3: 0.639674 (0.080567)
P4: 0.547283 (0.067172)
P5: 0.608424 (0.021065)
P6: 0.608424 (0.021065)
P7: 0.608424 (0.021065)

L1:
P8: 0.888859 (0.036467)
P9: 0.907880 (0.019385)
P10: 0.871739 (0.051398)
P11: 0.837228 (0.086330)
P12: 0.669293 (0.123473)
P13: 0.672283 (0.083492)
P14: 0.561413 (0.054797)
P15: 0.543750 (0.101417)
Best model is:
Perceptron(alpha=0.0001, class_weight=None, eta0=1.0, fit_intercept=True,
      max_iter=5000, n_iter=None, n_jobs=1, penalty='l1', random_state=0,
      shuffle=True, tol=None, verbose=0, warm_start=False)
With an accuracy score of: 0.8805646036916395
Took 5000 iterations
[[538  11]
 [ 99 273]]
             precision    recall  f1-score   support

        0.0       0.84      0.98      0.91       549
        1.0       0.96      0.73      0.83       372

avg / total       0.89      0.88      0.88       921
```

For neural networks:
```
NN0 identity: 0.889402 (0.029719)
NN1 identity: 0.889674 (0.029070)
NN2 identity: 0.897554 (0.018190)
NN3 identity: 0.897554 (0.018068)
NN4 identity: 0.897283 (0.018017)
NN5 identity: 0.883152 (0.030864)
NN6 identity: 0.851630 (0.048400)
NN7 logistic: 0.933696 (0.011981)
NN8 logistic: 0.932880 (0.012978)
NN9 logistic: 0.933152 (0.014844)
NN10 logistic: 0.934239 (0.012381)
NN11 logistic: 0.930435 (0.011348)
NN12 logistic: 0.908696 (0.011217)
NN13 logistic: 0.795109 (0.018518)
NN14 tanh: 0.936957 (0.014369)
NN15 tanh: 0.934239 (0.014420)
NN16 tanh: 0.937500 (0.012688)
NN17 tanh: 0.936141 (0.013869)
NN18 tanh: 0.933967 (0.014123)
NN19 tanh: 0.922011 (0.010736)
NN20 tanh: 0.877989 (0.014606)
NN21 relu: 0.892120 (0.031245)
NN22 relu: 0.901630 (0.042374)
NN23 relu: 0.914674 (0.016493)
NN24 relu: 0.916576 (0.024062)
NN25 relu: 0.923370 (0.012261)
NN26 relu: 0.904891 (0.021942)
NN27 relu: 0.875815 (0.023251)
Best model is:
MLPClassifier(activation='tanh', alpha=0.001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(100,), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=0, shuffle=True,
       solver='adam', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)
With an accuracy score of: 0.9283387622149837
[[513  36]
 [ 30 342]]
             precision    recall  f1-score   support

        0.0       0.94      0.93      0.94       549
        1.0       0.90      0.92      0.91       372

avg / total       0.93      0.93      0.93       921

Nr of iterations: 55
```