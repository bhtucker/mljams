# -*- coding: utf-8 -*-
"""
    matrix_factorization.py
    ~~~~~~~~~~~~~~~~~~~~~~~

    clustering & recommendation via matrix factorization
"""

import numpy as np
from scipy import stats
from adaboost_cancer import weighted_draw
from operator import add


def read_movie_data():
    with open("movies_csv/ratings.txt", "r") as f:
        ratings = map(lambda e: map(int, e.split(',')), f.readlines())
    with open("movies_csv/ratings_test.txt", "r") as f:
        ratings_test = map(lambda e: map(int, e.split(',')), f.readlines())
    return (ratings, ratings_test)


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
        self.objective_trace = []

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
        self.objective_trace.append(
            self.score_objective_fxn())

    def score_objective_fxn(self):
        score = 0.
        for i in range(self.n):
            score += np.linalg.norm(
                self.data[i] - self.mu[self.c[i]])
        return score


class CollaborativeFilteringFactorizer(object):
    """
    Factorize a N1xN2 data matrix with missing values into
    two matrices U (N1xd) and V (dxN2) s.t. M ~ U*V
    Probabilistically, find MAP solution for U and V
    supposing Mij is drawn from N(u_iT * v_j, sigmasq)
    and both u_i, v_j ~ N(0, lambda^-1 * I)
    Basically, performs coordinate ascent via ridge regression
    to alternate between improving U and V across iterations
    """
    def __init__(self, obs, d=20, sigmasq=.25, _lambda=10):
        # the set of observed (non-missing) elements of M
        self.obs = obs
        # the dimension of the factorization
        self.d = d
        # model noise parameter on M
        self.sigmasq = sigmasq
        # u_i, v_j prior variance size
        # aka regularization penalty
        self._lambda = _lambda
        self.n1 = max([x[0] for x in obs])
        self.n2 = max([x[1] for x in obs])
        self.u_i = {i: [] for i in range(1, self.n1 + 1)}
        self.v_j = {i: [] for i in range(1, self.n2 + 1)}
        # data structures for accessing the observed data
        self.user_objects = {
            i: set([x[1] for x in obs if x[0] == i])
            for i in range(1, self.n1 + 1)}
        self.object_users = {
            i: set([x[0] for x in obs if x[1] == i])
            for i in range(1, self.n2 + 1)}
        self.observed_i_j = {i: {} for i in range(1, self.n1 + 1)}
        for score in obs:
            self.observed_i_j[score[0]][score[1]] = score[2]
        self.rmse_trace = []
        self.log_likelihood_trace = []

    def learn(self, iterations, test_set=None):
        v_j_prior = stats.multivariate_normal(
            np.zeros(self.d), np.eye(self.d) * (1./self._lambda))
        # initialize v_j with samples from prior
        self.v_j = {
            ix + 1: np.transpose(np.matrix(val))
            for ix, val in enumerate(v_j_prior.rvs(self.n2))
        }
        # initialize the diagonal noise matrix used in updates
        self.lambda_sigmasq_eye = np.matrix(
            self._lambda * self.sigmasq * np.eye(self.d))

        for i in range(iterations):
            self.iterate(test_set=test_set)

    def update_locations(self, side='u'):
        # local vars are named as they would be in the default case
        # where U is being updated
        if side == 'u':
            to_update = self.u_i
            subset_map = self.user_objects
            to_read = self.v_j
        else:
            to_update = self.v_j
            subset_map = self.object_users
            to_read = self.u_i

        for i in to_update:
            obs_j_subset = subset_map[i]
            if not obs_j_subset:
                continue
            v_j_outer_products = [
                to_read[j] * np.transpose(to_read[j])
                for j in obs_j_subset]
            v_j_product_sum = reduce(add, v_j_outer_products)
            inverse_term = np.linalg.inv(
                self.lambda_sigmasq_eye + v_j_product_sum)
            if side == 'u':
                m_ij_v_j_products = [
                    self.observed_i_j[i][j] * to_read[j]
                    for j in obs_j_subset
                ]
            else:
                m_ij_v_j_products = [
                    self.observed_i_j[j][i] * to_read[j]
                    for j in obs_j_subset
                ]
            user_information_term = reduce(add, m_ij_v_j_products)
            if inverse_term.shape[1] != user_information_term.shape[0]:
                user_information_term = np.transpose(user_information_term)
            to_update[i] = np.matrix(inverse_term) * user_information_term

    def iterate(self, test_set=None):
        self.update_locations(side='u')
        self.update_locations(side='v')
        if test_set:
            self.rmse_trace.append(self.rmse(test_set))
        self.log_likelihood_trace.append(self.log_likelihood())

    def predict(self):
        return np.matrix(
            [[self.predict_element(i, j)
              for j in range(1, self.n2 + 1)]
             for i in range(1, self.n1 + 1)])

    def predict_element(self, i, j):
        rounded = round(
            float(np.transpose(self.u_i[i]) * self.v_j[j]))
        if rounded < 1:
            return 1.
        if rounded > 5:
            return 5.
        return rounded

    def rmse(self, test_data):
        return _RMSE([
            r[2] - self.predict_element(r[0], r[1])
            for r in test_data])

    def log_likelihood(self, data=None):
        data = data or self.obs
        data_term = np.product([
            normal_pdf(
                float(np.transpose(self.u_i[row[0]]) * self.v_j[row[1]]),
                self.sigmasq,
                [row[2]]
            )
            for row in data])

        u_v_terms = normal_pdf(
            np.zeros(self.d),
            np.matrix(self._lambda * np.eye(self.d)),
            self.u_i.values() + self.v_j.values())
        return data_term * u_v_terms


def _RMSE(error_vec):
    n = len(error_vec)
    return float(
        np.sqrt((1. / n) * sum([e ** 2 for e in error_vec]))
    )


def normal_pdf(mean, cov, data):
    if isinstance(mean, float):
        gauss = stats.norm(mean, cov)
        return np.product([gauss.pdf(datum) for datum in data])
    else:
        mvgauss = stats.multivariate_normal(mean, cov)
        return np.product([mvgauss.pdf(datum) for datum in data])
