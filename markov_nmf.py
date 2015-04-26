# -*- coding: utf-8 -*-
"""
    markov_nmf.py
    ~~~~~~~~~~~~~

    Markov abuses and NMF uses
"""
import numpy as np
import heapq


def _update_team(M, this_team, other_team):
    [j1, s1, j2, s2] = list(this_team) + list(other_team)
    M[j1][j1] += 1 if s1 > s2 else 0 + (s1 / (s1 + s2))
    M[j2][j1] += 1 if s1 > s2 else 0 + (s1 / (s1 + s2))
    return M


def generate_transition_matrix(scores, size=759):
    M = np.zeros((size, size))
    for [j1, s1, j2, s2] in scores:
        M = _update_team(M, (j1 - 1, s1), (j2 - 1, s2))
        M = _update_team(M, (j2 - 1, s2), (j1 - 1, s1))
    row_sums = M.sum(axis=1)
    normalized_M = M / row_sums[:, np.newaxis]
    return normalized_M


def get_w_t(M, t):
    (n, n) = M.shape
    w_t = np.ones(n) * (1. / n)
    for i in range(t):
        w_t = np.dot(w_t, M)
    return w_t

offset = 10**-16


def get_n_largest_indices(array, n):
    return heapq.nlargest(n, range(len(array)), array.take)


class NonnegativeMatrixFactorizer(object):
    """
    Create two rank k matrices W and H whose product approximates the matrix M.
    Optimize W and H according to penalty='squared_error' or penalty='divergence'
    "W is n × K, H is K × m"
    """
    def __init__(self, penalty='squared_error'):
        self.penalty = penalty

    def get_offset(self):
        return np.ones(self.M.shape) * offset

    def squared_error(self):
        self.H = np.matrix(
            np.array(self.H) * np.array(self.W.T * self.M) / np.array(self.W.T * self.W * self.H))
        self.W = np.matrix(
            np.array(self.W) * np.array(self.M * self.H.T) / np.array(self.W * self.H * self.H.T))
        self.objective_trace.append(np.linalg.norm(self.M - (self.W * self.H), ord=2))

    def divergence(self):
        
        self.H = np.matrix(
            np.array(self.H) \
            * np.array(
                row_normalize(self.W.T) *
                np.matrix(np.array(self.M) / (np.array(self.W * self.H) + self.get_offset())
            )))
        self.W = np.matrix(
            np.array(self.W)
            * np.array(
                np.matrix(np.array(self.M) / (np.array(self.W * self.H)  + self.get_offset())) *
                col_normalize(self.H.T)
            ))

        WH = self.W * self.H
        rv = np.array(self.M) * np.array(
            np.log(np.ones(WH.shape) / (WH + self.get_offset())))
        self.objective_trace.append(np.sum(rv+WH))


    def learn(self, M, k, iterations):
        (n, m) = M.shape
        self.M = M
        self.objective_trace = []
        self.W = np.matrix(np.random.exponential(np.mean(M), n * k).reshape((n, k)))
        self.H = np.matrix(np.random.exponential(np.mean(M), m * k).reshape((k, m)))
        iterate = getattr(self, self.penalty)
        for i in range(iterations):
            iterate()


def row_normalize(M):
    if isinstance(M, np.matrix):
        convert = True
        M = np.array(M)
    row_sums = M.sum(axis=1)
    normalized_M = M / row_sums[:, np.newaxis]
    if convert:
        return np.matrix(normalized_M)
    return normalized_M


def col_normalize(M):
    if isinstance(M, np.matrix):
        convert = True
        M = np.array(M)
    col_sums = M.sum(axis=0)
    normalized_M = M / col_sums[np.newaxis, :]
    if convert:
        return np.matrix(normalized_M)
    return normalized_M
