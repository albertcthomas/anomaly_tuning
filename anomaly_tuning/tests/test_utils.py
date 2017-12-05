# The tests for the GaussianMixture object are inspired from the tests of
# the GaussianMixture estimator implemented in scikit-learn.

import numpy as np

from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_array_almost_equal

from anomaly_tuning.utils import GaussianMixture


def test_sample():
    """Check sample from Gaussian Mixture."""
    random_state = 5
    rng = np.random.RandomState(random_state)

    n_features = 2
    weight_1 = rng.rand(1)[0]
    weight_2 = 1 - weight_1
    mean_1 = rng.rand(n_features)
    mean_2 = (rng.rand(n_features) + 2)
    A_1 = rng.rand(n_features, n_features)
    cov_1 = np.dot(A_1.T, A_1) + np.identity(n_features)
    A_2 = rng.rand(n_features, n_features)
    cov_2 = np.dot(A_2.T, A_2) + 0.5 * np.identity(n_features)
    weights = np.array([weight_1, weight_2])
    means = np.array([mean_1, mean_2])
    covars = np.array([cov_1, cov_2])

    gm = GaussianMixture(weights, means, covars, return_labels=True,
                         random_state=random_state)
    n_samples = 20000
    X_s, y_s = gm.sample(n_samples)
    # check shape
    assert_equal(X_s.shape, (n_samples, n_features))

    # check means, covariances and weights
    emp_covs = np.empty((2, n_features, n_features))
    for k in range(2):  # loop over number of components of the mixture
        emp_covs[k, ...] = np.cov(X_s[y_s == k].T)

    emp_means = np.array([np.mean(X_s[y_s == k], 0)
                          for k in range(2)])

    assert_array_almost_equal(covars, emp_covs, decimal=1)
    assert_array_almost_equal(means, emp_means, decimal=1)
    assert_almost_equal(np.mean(y_s == 0), weight_1, decimal=1)

    # check we get same X when return_labels=False
    gm_2 = GaussianMixture(weights, means, covars, random_state=random_state)
    n_samples = 20000
    X_s_2 = gm_2.sample(n_samples)
    assert_array_equal(X_s, X_s_2)
