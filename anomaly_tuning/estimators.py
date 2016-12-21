# Authors: Albert Thomas
#          Alexandre Gramfort
# License: BSD (3-clause)

# Anomaly detection estimators used in the papers. Except for the ones based
# on k-NN (AverageKLPE, MaxKLPE), the others are implemented in scikit-learn.

import numpy as np

from sklearn import ensemble
from sklearn.base import BaseEstimator
from sklearn.svm import OneClassSVM
from sklearn.neighbors import NearestNeighbors, KernelDensity


class KLPE(BaseEstimator):
    """Anomaly detection scoring functions based on k nearest neighbors.

    Returns a scoring function that rank observations according to their
    degree of abnormality.

    The implementation is based on the scikit-learn implementation of nearest
    neighbors.

    Parameters
    ----------
    k : integer, optional (default=6)
        Number of neighbors to consider

    algo : string, optional (default='average')
        Specify how the scoring function is estimated. If algo is 'max', the
        score of x is the distance of x to its k-th nearest neighbor in the
        data set X_train.
        If algo is 'average', the score of x is the average of its distances
        to its k-th nearest neighbors in the data set X_train.

    threshold : float, optional (default=0.05)
        Threshold to declare wether an observation is abnormal or not. An
        observation is declared abnormal if its score belongs to the
        (threshold * 100)% lowest scores. Should be in (0, 1).

    References
    ----------
    .. [1] Zhao, M. and Saligrama, V. "Anomaly Detection with Score functions
        based on Nearest Neighbor Graphs". In Advances in Neural Information
        Processing Systems 22. 2009.
    .. [2] Qian, J. and Saligrama, V. "New statistic in p-value estimation for
        anomaly detection". In Statistical Signal Processing Workshop (SSP),
        2012 IEEE. 2012.
    """

    name = 'klpe'

    def __init__(self, k=6, algo='average', threshold=0.05):

        self.k = k
        self.algo = algo
        self.threshold = threshold

    def fit(self, X_train):
        """For each observation in X_train, computes the distance of the k+1
        nearest neighbors in X_train.

        Parameters
        ----------
        X_train : array, shape (n_samples, n_features)
            Training data set.

        """

        k = self.k

        n_samples, n_features = X_train.shape

        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X_train)
        dist, _ = nbrs.kneighbors(X_train)

        self.nbrs_ = nbrs
        self.X_train_ = X_train
        self.dist_train_ = dist

        return self

    def _get_dist(self, X):
        """Computes the distances of each observation in the data set X to its
        neighbors in X_train.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Data set.

        Returns
        -------
        dists : array, shape (n_samples,)
            Distances of the samples X to their neighbors in X_train.

        """

        k = self.k
        nbrs = self.nbrs_
        X_train = self.X_train_
        dist_train = self.dist_train_

        n_samples, n_features = np.shape(X)

        dists = dist_train[:, 1:]  # exclude the point itself

        if not np.array_equal(X_train, X):
            dists, _ = nbrs.kneighbors(X, n_neighbors=k)

        return dists

    def score_samples(self, X):
        """Computes the score of each observation in the data set X with the
        convention of our paper: the smaller the score the more abnormal the
        observation.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Data set.

        Returns
        -------
        score : array, shape (n_samples,)
            Returns the scoring function of the samples.

        """

        algo = self.algo

        dists = self._get_dist(X)

        if algo == 'max':
            score = - np.max(dists, axis=1)  # - because of convention
        elif algo == 'average':
            score = - np.mean(dists, axis=1)  # - because of convention

        return score

    def predict(self, X):
        """Predict class for X : +1 if normal, 0 if abnormal. Respects
        scikit-learn convention "the higher, the better".

        The anomalies are the observations whose scores belong to the
        (threshold * 100)% lowest scores of the training set.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Data set.

        Returns
        -------
        Y : array, shape (n_samples,)
            Predicted classes.

        """

        n_samples, _ = X.shape

        dists_train = self._get_dist(self.X_train_)
        dists_test = self._get_dist(X)

        proba = np.zeros(n_samples)
        for i in range(n_samples):
            proba[i] = np.mean(dists_train >= dists_test[i])

        Y = proba >= self.threshold

        return Y.astype(int)


class AverageKLPE(KLPE):
    """Anomaly detection estimator based on KLPE estimator with algo='average'

    Parameters
    ----------
    k : integer, optional (default=6)
        Number of neighbors to consider

    """

    name = 'aklpe'

    def __init__(self, k=6):
        self.k = k
        self.algo = 'average'


class MaxKLPE(KLPE):
    """Anomaly detection estimator based on KLPE estimator with algo='max'

    Parameters
    ----------
    k : integer, optional (default=6)
        Number of neighbors to consider

    """

    name = 'mklpe'

    def __init__(self, k=6):
        self.k = k
        self.algo = 'max'


class OCSVM(OneClassSVM):
    """Anomaly detection estimator based on the One-Class SVM. The kernel used
    is the Gaussian kernel defined as
                k(x, x') = exp(- norm(x-x',2)^2 / (2 * sigma^2))

    Parameters
    ----------
    sigma : float, optional (default=1.0)
        Gaussian kernel bandwidth. Should be > 0.

    nu : float, optional (default=0.4)
        nu parameter of the OneClassSVM algorithm. Should be in (0, 1].

    """

    name = 'ocsvm'

    def __init__(self, sigma=1.0, nu=0.4):
        gamma = 1. / (2. * sigma ** 2)
        OneClassSVM.__init__(self, gamma=gamma, nu=nu)

    def score_samples(self, X):
        """Scoring function of the estimator.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Input data set.

        Returns
        -------
        score : array, shape (n_samples,)
            Returns scores of the samples.

        """

        return self.decision_function(X).ravel()


class IsolationForest(ensemble.IsolationForest):
    """Anomaly detection estimator based on Isolation Forest."""

    name = 'iforest'

    def score_samples(self, X):
        """ Scoring function of the estimator.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Input data set.

        Returns
        -------
        score : array, shape (n_samples,)
            Returns scores of the samples.

        """

        return self.decision_function(X)


class KernelSmoothing(KernelDensity):
    """Anomaly detection estimator based on plug-in approach."""

    name = 'ks'
