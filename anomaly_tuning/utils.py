# Authors: Albert Thomas
#          Alexandre Gramfort
# License: BSD (3-clause)

import numpy as np

from scipy.stats import multivariate_normal

from sklearn.utils import check_random_state

from joblib import Parallel, delayed


def _parallel_score_samples(estimator, X):
    """Compute score_samples of a fitted estimator on X.

    This function is used to parallelize the computation of the scores of an
    ensemble.
    """
    return estimator.score_samples(X)


def compute_ensemble_score_samples(models, X, n_jobs=1):
    """Compue the score of each sample obtained with the ensemble

    Parameters
    ----------
    models : list, shape (n_estimator)
        Ensemble of anomaly detection estimators, each of them instantiated
        with the best hyperparameters learnt from their corresponding random
        split and fitted on X_train.
    X : array, shape (n_samples, n_features)
        Data set.
    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel to compute the scores
        If -1, then the number of jobs is set to the number of cores.

    Returns
    -------
    score : array, shape (n_samples,)
        Scores of the samples.
    """
    n_estimators = len(models)

    y_scores = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(_parallel_score_samples)(models[n_est], X)
        for n_est in range(n_estimators))

    score = np.mean(y_scores, axis=0)

    return score


class GaussianMixture(object):
    """Gaussian mixture.

    This class is inspired from scikit-learn GaussianMixture module to
    sample from and compute the density of a Gaussian mixture model.

    Parameters
    ----------
    weights : array, shape (n_components, )
        Weights of the Gaussian mixture components.

    means : array, shape (n_components, n_features)
        Means of the Gaussian mixture components.

    covars : array, shape (n_components, n_features, n_features)
        Covariances of the Gaussian mixture components.

    return_labels : boolean, optional (default=False)
        Whether to return component labels when sampling from the Gaussian
        Mixture. These labels are useful to test the `sample` method.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """

    def __init__(self, weights, means, covars, return_labels=False,
                 random_state=42):

        self.weights = weights
        self.means = means
        self.covars = covars
        self.return_labels = return_labels
        self.random_state = random_state

    def sample(self, n_samples):
        """Generates samples from the Gaussian Mixture.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to generate. Defaults to 1.

        Returns
        -------
        X : array, shape (n_samples, n_features)
            Randomly generated sample

        y : array, shape (nsamples,)
            Component labels. The component labels are computed and returned
            for the sake of testing when the attribute `return_labels` is set
            to True.
        """

        weights = self.weights
        means = self.means
        covars = self.covars
        random_state = self.random_state

        rng = check_random_state(random_state)
        n_samples_comp = rng.multinomial(n_samples, weights)

        X = np.vstack([
            rng.multivariate_normal(mean, cov, int(sample))
            for (mean, cov, sample) in zip(
                means, covars, n_samples_comp)])

        if self.return_labels:
            y = np.concatenate([j * np.ones(sample, dtype=int)
                                for j, sample in enumerate(n_samples_comp)])
            return X, y

        return X

    def density(self, X):
        """Gaussian Mixture density of the samples X."""

        weights = self.weights
        means = self.means
        covars = self.covars

        n_samples, _ = X.shape
        density = np.zeros(n_samples)

        for (weight, mean, cov) in zip(weights, means, covars):
            density += weight * multivariate_normal.pdf(X, mean, cov)

        return density
