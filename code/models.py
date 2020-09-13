# -*- coding: utf-8 -*-
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import binarize as bin
from data import NEGATIVE, POSITIVE

class BernoulliNaiveBayes(BaseEstimator, ClassifierMixin):
    """
    Bernoulli Naive Bayes classifier with 2 classes.
    """
    def __init__(self, binarize=.0, k=1):
        """
        Set:
            k value for Laplacian Smoothing
            binarize threshold: values = 1 if values > threshold else 0
        """
        self.binarize = binarize
        self.k = k

    def fit(self, X, y):
        """
        Input:
            X: n*m csr_matrix (sparse matrix)
            y: list of length n
        """
        k = self.k
        n, m = X.shape
        X = bin(X, threshold=self.binarize)
        num_y_1 = np.sum(y)
        num_y_0 = n - num_y_1

        """
        Define
            theta_1 = (# of examples where y=1) / (total # of examples)
            theta_j_1 = (# examples with xj=1 and y=1) / (# examples with y=1)
            theta_j_0 = (# examples with xj=1 and y=0) / (# examples with y=0)
        Then
            theta_x_1[j] = theta_j_1
            theta_x_0[j] = theta_j_0
        """
        theta_1 = num_y_1 / n
        theta_x_0 = np.full(m, k)
        theta_x_1 = np.full(m, k)

        for i in range(n):
            if y[i] == NEGATIVE:
                theta_x_0 += X[i]

            else: # y[i] == POSITIVE
                theta_x_1 += X[i]

        theta_x_0 = theta_x_0 / (num_y_0 + k + 1)
        theta_x_1 = theta_x_1 / (num_y_1 + k + 1)

        ones = np.full(m,1)

        """
        Define
            w_j_0 = log ((1 - theta_j_1) / (1 - theta_j_0))
            w_j_1 = log (theta_j_1 / theta_j_0)
        Then
            w_x_0[j] = x_j_0
            w_x_1[j] = x_j_1
        """
        w_x_0 = np.log(ones - theta_x_1) - np.log(ones - theta_x_0)
        w_x_1 = np.log(theta_x_1) - np.log(theta_x_0)

        """
        Define
            w_0 = log (P(y=1) / P(y=0)) + sum of w_j_0 for all j
            w = w_x_1 - w_x_0

        Then, for a given datapoint x, the log-odds ratio is:
            w_0 + (x.transpose * w)
        """
        w_0 = np.log(theta_1/(1 - theta_1)) + np.sum(w_x_0)
        w = w_x_1 - w_x_0

        self.w_0 = w_0
        self.w = w

        return self

    def predict(self, X):
        """
        Closed form solution for decision boundary.
        """
        n = X.shape[0]
        X = bin(X, threshold=self.binarize)
        w_0 = self.w_0
        w = self.w
        y_pred = np.full(n, w_0)

        """
        y_pred[i] is the log-odds ratio of datapoint x_i
        """
        for i, x in enumerate(X):
            y_pred[i] += x.dot(w.T)

        """
        If the log-odds ratio of datapoint x_i >= 0,
        then y_pred[i] = POSITIVE
        """
        y_pred = (y_pred >= 0).astype(int).tolist()

        return y_pred
