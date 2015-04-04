# -*- coding: utf-8 -*-
"""
    matrix_factorization.py
    ~~~~~~~~~~~~~~~~~~~~~~~

    clustering & recommendation via matrix factorization
"""

import numpy as np
from scipy import stats
from adaboost_cancer import weighted_draw


def generative_model(size):
    """
    Draw data from a mixture of three multivariate gaussians
    """
    prevalence = [0.2, 0.5, 0.3]
    means = [
        [0, 0],
        [3, 0],
        [0, 3]
    ]

    covs = [[[1, 0], [0, 1]]] * 3

    gaussians = [
        stats.multivariate_normal(m, c)
        for (m, c) in zip(means, covs)
    ]

    sample_gaussian_indices = weighted_draw(prevalence, size)

    return [gaussians[i].rvs(1) for i in sample_gaussian_indices]


class KMeansClusterMachine(object):
    """
    Hard clustering algorithm finding K centroids that minimize
    the distance from centroids to the 'cluster' of points nearest each centroid.
    """
    def __init__(self, data):
        self.data = data if isinstance(data, np.ndarray) else np.array(data)
        self.n = len(data)
        self.c = np.zeros(self.n)
        self.c_trace = []
        self.mu_trace = []

    def update_c(self):
        """
        Generate cluster assignments for each point
        by finding nearest centroid in self.mu
        """
        mu_vecs = [
            np.array([mu_k] * self.n)
            for mu_k in self.mu]
        data_distances = [
            np.linalg.norm(self.data - mu_vec, axis=1)
            for mu_vec in mu_vecs]
        self.c = np.array([
            np.argmin([x[i] for x in data_distances])
            for i in range(self.n)]
        )

    def update_mu(self):
        """
        Generate centroids by calculating mean
        from current cluster assignments
        """
        self.mu = []
        for i in range(self.k):
            (idx, ) = np.where(self.c == i)
            self.mu.append(
                np.mean([
                    self.data[i]
                    for i in idx], axis=0))
        self.mu = np.array(self.mu)

    def learn(self, k, iterations):
        """
        Initialize centroids randomly
        then update c and mu iterations times
        """
        self.k = k
        self.mu = np.array([
            self.data[i]
            for i in np.random.randint(0, self.n, self.k)])
        for i in range(iterations):
            self.iterate()

    def iterate(self):
        """
        Make updates and save step values
        """
        self.c_trace.append(self.c)
        self.update_c()
        self.mu_trace.append(self.mu)
        self.update_mu()
