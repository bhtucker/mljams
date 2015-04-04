# -*- coding: utf-8 -*-
"""
    cancer_adaboost.py
    ~~~

    code for adaboost cancer hw
"""

import numpy as np
from numpy.linalg import LinAlgError


def split_data(X, Y, test=183):
    """
    Returns test_x, test_y, train_x, train_y
    """
    data = zip(X, Y)
    return (
        [x[0] for x in data[:test]],
        [x[1] for x in data[:test]],
        [x[0] for x in data[test:]],
        [x[1] for x in data[test:]])


def get_accuracy(classifier, x, y, predictions=None):
    predictions = _get_prediction_vector(classifier, x)
    predictions, y = np.array(predictions), np.array(y)
    return get_prediction_accuracy(predictions, y)


def get_prediction_accuracy(predictions, y):
    return len(np.where((predictions - y) == 0)[0]) / float(len(y))


def weighted_draw(w, n, verify=False):
    """
    Accept a positive integer n and a discrete, k-dimensional probability distribution w,
    Return a 1 × n vector c, where ci ∈ {1, . . . , k},
    Prob(ci = j|w) = w(j) and the entries of c are independent
    """
    cutoffs = np.array([
        w_element + sum(w[:idx])
        for idx, w_element in enumerate(w)])

    if verify:
        # verify that w is a proability distribution
        assert sum(w) == 1
        # verify that w is reproduced from cutoffs
        assert abs(sum(
            np.array([
                cutoffs[i] - cutoffs[i-1] if i > 0 else cutoffs[i]
                for i in range(len(cutoffs))]
            ) - np.array(w))) < 1e-15

    def _bucket(random_draw):
        # give each dimension a region of the interval [0, 1] proportional to its weight
        # broadcast subtraction to find region a random draw falls
        return min(np.where((np.array(cutoffs) - random_draw) > 0)[0])

    return map(_bucket, np.random.uniform(size=n))


def calculate_weighted_error(predictions, y_train, w):
    """
    Iterate over training set, incrementing error score
    by weight for each misclassified example
    """
    assert len(predictions) == len(w) == len(y_train), \
        'weights and training data must have same length'
    error_score = 0.
    for idx, pred in enumerate(predictions):
        if y_train[idx] != pred:
            error_score += w[idx]
    assert error_score > 0., 'perfect weak classifier! ensemble method asplode!'
    return error_score


def calculate_voter_accuracy(eps):
    """
    Get weighted error, calculate and return alpha
    """
    return .5 * np.log((1. + eps) / eps)


def update_w(predictions, y_train, w, alpha):
    """
    For each observation i in training data, wt+1(i) =
        wt(i) * exp ( -alpha t * true_label(i) * predictiont(i))
    Normalize new w before returning:
        wt+1(i) = ˜wt+1(i)/Pj w˜t+1(j)
    """
    new_w = []
    for idx, pred in enumerate(predictions):
        updated_w_element = w[idx] * \
            np.exp(-1 * alpha * y_train[idx] * pred)
        new_w.append(updated_w_element)

    new_w = np.array(new_w)
    return new_w * (1. / sum(new_w))


def _get_prediction_vector(classifier, x_train):
    assert hasattr(classifier, 'classify'), \
        'expect classifier to implement classify method'
    return map(classifier.classify, x_train)


class OnlineLogisticClassifier(object):
    """
    A linear classifier updated with each data point it sees
    'likelihood' of class yi for observation xi is given by
            1/(1 + e(−yi xTiw))
    """
    def __init__(self, learning_rate):
        super(OnlineLogisticClassifier, self).__init__()
        self.w = np.array([])
        self.learning_rate = learning_rate

    def _update_w(self, xi, yi):
        if len(self.w) != len(xi):
            assert not self.w
            self.w = np.zeros(len(xi))
        self.w += self.learning_rate * (
            (1. - self.sigmoid(xi, yi)) * (yi * xi)
        )

    def sigmoid(self, xi, yi):
        return 1. / (1. +
                     np.exp(-1 * yi * np.dot(xi, self.w)))

    def classify(self, xi):
        if self.sigmoid(xi, 1) < .5:
            return -1
        else:
            return 1

    def train(self, x, y):
        for (xi, yi) in zip(x, y):
            self._update_w(xi, yi)


class AdaBoostClassifier(object):
    """
    Implements adaptive boosting on an arbitrary classifier
    Construct with a classifier class that implements:
        .classify(Xi)
        .train(X, Y)
    Pass kwargs for 'child' classifier to constructor
    """
    def __init__(self, classifier, **kwargs):
        self.classifier = classifier
        self.alpha_vec = []
        self.eps_vec = []
        self.w_vec = []
        self.classifiers = []
        self.data_size = 0
        self.classifier_kwargs = kwargs

    def iterate(self):
        w = self.w_vec[-1]

        def sample_and_learn():
            sample_indices = weighted_draw(w, self.data_size)
            sample_X = [self.X[i] for i in sample_indices]
            sample_Y = [self.Y[i] for i in sample_indices]
            step_classifier = self.classifier(**self.classifier_kwargs)
            try:
                step_classifier.train(sample_X, sample_Y)
            except LinAlgError:
                # in case you draw a non-invertible sample covariance
                sample_and_learn()
            return step_classifier

        step_classifier = sample_and_learn()
        predictions = _get_prediction_vector(step_classifier, self.X)
        eps = calculate_weighted_error(predictions, self.Y, w)

        alpha = calculate_voter_accuracy(eps)
        new_w = update_w(predictions, self.Y, w, alpha)
        self.alpha_vec.append(alpha)
        self.eps_vec.append(eps)
        self.w_vec.append(new_w)
        self.classifiers.append(step_classifier)

    def learn_ensemble(self, X, Y, iterations):
        self.data_size = len(Y)
        self.w_vec.append(np.array([1. / self.data_size] * self.data_size))
        self.X = X
        self.Y = Y
        for t in range(iterations):
            self.iterate()

    def classify(self, new_x):
        assert len(self.classifiers) == len(self.alpha_vec)
        committee_results = [
            self.classifiers[i].classify(new_x) * self.alpha_vec[i]
            for i in range(len(self.classifiers))]
        return 1 if sum(committee_results) > 0 else -1


def ensure_nparray(*args):
    return [x if isinstance(x, np.ndarray) else np.array(x) for x in args]


class SharedCovarianceBayesClassifier(object):
    """Using a shared covariance matrix and class-specific
    prevalence and means, classify data through linear discrimnant analysis"""
    def __init__(self):
        pass

    def train(self, x, y):
        n = len(x)
        [y] = ensure_nparray(y)
        positive_indices = set(np.where(y > 0)[0])
        if all([x_element[0] == 1 for x_element in x]):
            x_strip = map(lambda r: r[1:], x)
        else:
            x_strip = x
        negative_indices = set(range(n)) - positive_indices
        positive_prev = len(positive_indices) / float(n)
        negative_prev = len(negative_indices) / float(n)

        positive_mean = np.mean([x_strip[i] for i in positive_indices], axis=0)
        negative_mean = np.mean([x_strip[i] for i in negative_indices], axis=0)
        shared_cov = np.cov(x_strip, bias=True, rowvar=False)
        shared_precision = np.linalg.inv(shared_cov)
        w_0 = (
            np.log((positive_prev / negative_prev))
            - (
                .5 * np.dot(np.dot(
                    (positive_mean + negative_mean), shared_precision),
                    (positive_mean - negative_mean))
            ))

        w = np.dot(shared_precision, (positive_mean - negative_mean))

        self.w_aug = np.array([el for el in [w_0] + list(w)])

    def classify(self, new_x):
        return 1 if np.dot(self.w_aug, new_x) > 0 else -1


"""
Deprecated in favor of dynamic programming version below:
def learn_ensemble_with_trace(booster, iterations, x_train, y_train, x_test, y_test):
    \"""
    returns test_error_trace, train_error_trace
    \"""
    train_error_trace = []
    test_error_trace = []
    booster.learn_ensemble(x_train, y_train, 1)
    test_error_trace.append(1. - get_accuracy(booster, x_test, y_test))
    train_error_trace.append(1. - get_accuracy(booster, x_train, y_train))
    for i in range(iterations - 1):
        booster.iterate()
        test_error_trace.append(1. - get_accuracy(booster, x_test, y_test))
        train_error_trace.append(1. - get_accuracy(booster, x_train, y_train))
    return test_error_trace, train_error_trace
"""


def get_error_traces(trained_ensemble, x_train, y_train, x_test, y_test):
    """
    Returns test_error_trace, train_error_trace
    Construct predictions once and construct the ensemble vote at step t
    by using the linear combination of votes 0-t with coefs. alphas 0-t
    """
    weak_test_prediction_vecs = [
        _get_prediction_vector(trained_ensemble.classifiers[i], x_test)
        for i in range(len(trained_ensemble.classifiers))
    ]

    weak_train_prediction_vecs = [
        _get_prediction_vector(trained_ensemble.classifiers[i], x_train)
        for i in range(len(trained_ensemble.classifiers))
    ]

    def get_committee_prediction(prediction_matrix, ensemble, t):
        return [
            1 if sum(
                [prediction_matrix[i][p] * ensemble.alpha_vec[i]
                    for i in range(t)]
            ) > 0 else -1
            for p in range(len(prediction_matrix[0]))]

    test_error_trace, train_error_trace = [], []
    iterations = len(trained_ensemble.alpha_vec)
    y_train, y_test = ensure_nparray(y_train, y_test)
    for t in range(iterations):
        test_error_trace.append(
            1 - get_prediction_accuracy(
                np.array(get_committee_prediction(weak_test_prediction_vecs, trained_ensemble, t)),
                y_test))
        train_error_trace.append(
            1 - get_prediction_accuracy(
                np.array(get_committee_prediction(weak_train_prediction_vecs, trained_ensemble, t)),
                y_train))
    return test_error_trace, train_error_trace
