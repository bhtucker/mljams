# mljams
personal (if partial) implementations of major machine learning algos

### multiclass classifiers

* KNN - predict class by the nearest K observations,
* multivariate gaussian Bayes classifier - learned via maximum likelihood estimation
* softmax logistic regression classifiers - learned via gradient descent


### adaptive boosting
train an ensemble of weak classifiers


provably drives training error to zero (training error over iterations in blue)


and gives pretty solid generalizability (test error over iterations in green)


![Training & Test Error Traces](http://i.imgur.com/7qg1vCj.png)
