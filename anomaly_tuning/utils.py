# Authors: Albert Thomas
#          Alexandre Gramfort
# License: BSD (3-clause)

# Gaussian Mixture class inspired from scikit-learn GaussianMixture module to
# sample and compute density of a Gaussian mixture model.

import numpy as np

from scipy.stats import multivariate_normal


class GaussianMixture(object):
    """Gaussian mixture.

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
        Mixture. These labels are useful to test the sample method.

    random_state : int
        Seed used by the random number generator.
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
            for the sake of testing when the attribute return_labels is set to
            True.
        """

        weights = self.weights
        means = self.means
        covars = self.covars
        random_state = self.random_state

        rng = np.random.RandomState(random_state)
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
