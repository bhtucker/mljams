# -*- coding: utf-8 -*-
"""
    classifier.py
    ~~~~~~~~~~~~~
 
    Implementations of 
        KNN,
        multivariate gaussian Bayes learned via MLE, and
        softmax logistic regression classifiers learned via gradien descent
    Sketch-level stuff, not totally turn-key at this point

"""


from collections import Counter, defaultdict
import numpy as np


class KNNClassifier(object):
    """Evaluate a test data point relative to some training data
       and return the most probably class based on the nearest K points
       by Euclidean distance"""
    def __init__(self, k, x_train, y_train):

        self.k = k
        self.x_train = np.array(x_train)
        self.y_train = np.array(y_train)

    def classify(self, x_test):
        # find vectors from each training point to the test point
        diff_vecs = self.x_train - x_test
        # find norms of these difference vectors
        row_norms = np.linalg.norm(diff_vecs, axis=1)
        # let's find the vectors having distance <= cutoff
        neighbor_indices = row_norms.argpartition(self.k)[:self.k]
        votes = Counter(np.take(y_train, neighbor_indices))
        # From docs for .most_common():
        #  Elements with equal counts are ordered arbitrarily
        return votes.most_common(1)[0][0]


class ConfusionMatrix(object):
    """nested dictionaries for storing test values"""
    def __init__(self):
        super(ConfusionMatrix, self).__init__()
        self._data = defaultdict(lambda: defaultdict(lambda: 0))
        self._n = 0

    def update(self, true, pred):
        self._data[true][pred] += 1
        self._n += 1

    def accuracy(self):
        return sum([
            self._data[i].get(i, 0)
            for i in range(int(max(self._data.keys())))
        ]) * (1. / self._n)


def get_confusion_matrix(classifier, x_test, y_test):
    confusion_mat = ConfusionMatrix()
    errors = []
    for test_x, test_y in zip(x_test, y_test):
        pred = classifier.classify(test_x)
        if isinstance(test_y, list):
            true = test_y[0]
        else:
            true = test_y
        confusion_mat.update(true, pred)
        if true != pred:
            errors.append([test_x, true, pred])
    return confusion_mat, errors


def _fit_multivariate_normal(x):
    # From numpy docs for np.mean:
    #   The arithmetic mean is the sum of the elements along the axis divided by the number of elements.
    #   axis: Axis along which the means are computed
    # My interpretation and reason for use:
    # So we use np.mean to efficiently compute the mean 'vertically',
    # summing values for each dimension and dividing by the number of rows
    sample_mean = np.mean(x, axis=0)
    # From numpy docs for np.cov:
    #   If we examine N-dimensional samples, X = [x_1, x_2, ... x_N]^T, 
    #   then the covariance matrix element C_{ij} is the covariance 
    #   of x_i and x_j. The element C_{ii} is the variance of x_i.
    #   bias : int, optional
    #           Default normalization is by ``(N - 1)``, where ``N`` is the number of
    #           observations given (unbiased estimate). If `bias` is 1, then
    #           normalization is by ``N``.
    # My interpretation and reason for use:
    # So we can efficiently compute covariance by viewing the data matrix
    # as a collection of samples of each dimension, each stored on a column.
    # Construct a covariance matrix by finding the covariance between these column
    # samples, or, for the diagonal entries, the variance of each sample.
    # Because we are estimating by maximum likelihood, use bias=1
    # Essentially just calls (dot(X, X.T) / N).squeeze() with nice type checking
    sample_cov = np.cov(x, bias=True, rowvar=False)
    # Create an instance of scipy.stats.multivariate normal with these parameters
    # We'll just use its PDF method
    return stats.multivariate_normal(mean=sample_mean, cov=sample_cov)


class MultivariateNormalClassifier(object):
    """Evaluate a test data point relative to some training data
       by fitting a multivariate normal distribution to the training
       data of each class and assessing the density of each class conditional
       density function at the location of the test point."""
    def __init__(self, x_train, y_train):
        class_labels = set([y for x in y_train for y in x])
        self.x_train = np.array(x_train)
        self.y_train = np.array(y_train)
        # IN THIS CASE, we know in advance that all classes are present in equal number
        # in the training data, so we drop the p(class_k) term from the prediction
        self._gaussians = {
            label: _fit_multivariate_normal(
                np.array([x_train[i] for i in np.where(np.array(y_train) == label)[0]])
            )
            for label in class_labels
        }

    def classify(self, test_x, verbose=False):
        # use the pdf method from the multivariate_normal objects in self._gaussians
        # the meat of the implementation is:
        #   const * exp ( -0.5 * (
        #       rank * _LOG_2PI + log_det_cov
        #       + np.sum(np.square(np.dot(dev, prec_U)), axis=-1)))
        # where prec_U, rank, log_det_cov come from symmetric eigendecomposition of precision matrix
        # (pseudo-inverse of covariance matrix) and dev is the deviation of x from the mean
        densities = [
            (label, mvn.pdf(test_x))
            for label, mvn in self._gaussians.iteritems()
        ]
        densities.sort(key=lambda r: -r[1])
        if verbose:
            return densities[0][0], densities
        else:
            return densities[0][0]


def _gradient(k, w_container, x_train, y_train):
    n_grad = [
        np.array(train_x) * (
            ((1 if train_y[0] == k else 0) -
            (
                np.exp(np.dot(np.transpose(train_x), w_container[k])) /
                sum([
                    np.exp(np.dot(np.transpose(train_x), w_container[j])) 
                    for j in w_container
                    ])
                )
            ))
        for train_x, train_y in zip(x_train, y_train)
    ]

    return np.sum(n_grad, axis=0)


def _loglikelihood(w_container, x_train, y_train):
    n_log_likelihood = [
        np.dot(np.transpose(train_x), w_container[train_y[0]]) - 
        np.log(
                sum([
                    np.exp(np.dot(np.transpose(train_x), w_container[j])) 
                    for j in w_container
                    ])
            )
        for train_x, train_y in zip(x_train, y_train)
    ]
    return sum(n_log_likelihood)


def _iterate(w_container, x_train, y_train, nu=.1 / 5000):
    w_container = _update_w_container((w_container, x_train, y_train, nu))
    return w_container, _loglikelihood(w_container, x_train, y_train)


def get_likelihood_trace(w_container, x_train, y_train):
    """
    For plotting likelihood across iterations
    """
    likelihood_trace = []
    for i in range(1000):
        w_container, log_likelihood = _iterate(w_container, x_train, y_train)
        likelihood_trace.append(log_likelihood)
