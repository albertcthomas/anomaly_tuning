from __future__ import division

import numpy as np

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc
from sklearn.utils.testing import assert_true
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_equal

from anomaly_tuning.estimators import AverageKLPE
from anomaly_tuning.estimators import MaxKLPE
from anomaly_tuning.estimators import IsolationForest
from anomaly_tuning.estimators import KernelSmoothing
from anomaly_tuning.estimators import OCSVM
from anomaly_tuning.tuning import _compute_volumes
from anomaly_tuning.tuning import est_tuning
from anomaly_tuning.tuning import anomaly_tuning

algorithms = [AverageKLPE, MaxKLPE, OCSVM, IsolationForest, KernelSmoothing]
algo_param = {'aklpe': {'k': np.arange(1, 10, 2), 'novelty': [True]},
              'mklpe': {'k': np.arange(1, 10, 2), 'novelty': [True]},
              'ocsvm': {'sigma': np.linspace(0.01, 5., 5)},
              'iforest': {'max_samples': np.linspace(0.1, 1., 5),
                          'random_state': [42]},
              'ks': {'bandwidth': np.linspace(0.01, 5., 5)},
              }

rng = np.random.RandomState(42)
n_features = 2
X = rng.randn(50, n_features)
n_estimator = 2
cv = ShuffleSplit(n_splits=n_estimator, test_size=0.2, random_state=42)

X_train, X_test = train_test_split(X, test_size=0.2, random_state=23)

X_range = np.zeros((n_features, 2))
X_range[:, 0] = np.min(X, axis=0)
X_range[:, 1] = np.max(X, axis=0)
# volume of the hypercube enclosing the data
vol_tot_cube = np.prod(X_range[:, 1] - X_range[:, 0])

# hypercube sampling: sampling uniformly in X_range
n_sim = 100
# reseed generator to have same U in anomaly_tuning
rng = np.random.RandomState(42)
U = np.zeros((n_sim, n_features))
for l in range(n_features):
    U[:, l] = rng.uniform(X_range[l, 0], X_range[l, 1], n_sim)


def test_compute_volumes_toy():
    """Check _compute_volumes on a toy scoring function."""

    def score_function_toy(X):
        """ score(x_0, x_1) = x_0 """
        return X[:, 0]

    # regular grid with step size equal to 0.01
    xx, yy = np.meshgrid(np.arange(0, 1, 0.01),
                         np.arange(0, 1, 0.01))
    grid = np.c_[xx.ravel(), yy.ravel()]

    alphas = rng.randint(1, 100, size=5) / 100
    vol, offset = _compute_volumes(score_function_toy, alphas, grid, grid, 1.)
    assert_array_equal(alphas, vol)

    # TODO add test for offset when bisect is replaced by quantile


def test_compute_volumes():
    """Check _compute_volumes for several masses."""
    estimators = [AverageKLPE(k=3, novelty=True), MaxKLPE(k=3, novelty=True),
                  OCSVM(sigma=1.),
                  IsolationForest(n_estimators=5, random_state=2),
                  KernelSmoothing()]
    alphas = rng.randint(1, 100, size=5) / 100
    alphas = np.sort(alphas)

    for clf in estimators:
        clf = clf.fit(X_train)
        clf_test = clf.score_samples(X_test)
        min_test = np.min(clf_test)
        max_test = np.max(clf_test)

        score_function = clf.score_samples
        vols, offsets = _compute_volumes(score_function, alphas, X_test,
                                         U, vol_tot_cube)
        # check increasing order of volumes and decreasing order of offsets
        assert_array_equal(vols, np.sort(vols))
        assert_array_equal(offsets, -np.sort(-offsets))

        # check volumes in [0, vol_tot_cube]
        assert_true(np.all(0 <= vols) and np.all(vols <= vol_tot_cube))

        # check offset values
        for alpha, offset in zip(alphas, offsets):
            # TODO remove 1e-12 when quantile
            assert_true(min_test - 2e-12 <= offset <= max_test + 1e-12)
            # TODO when quantile
            # assert_true(np.mean(clf_test >= offset) <= alpha)

        # TODO add test for alpha=0 and 1 when bisect is replaced by quantile


def test_est_tuning():
    """Check that est_tuning returns the estimator with minimum auc."""

    for algo in algorithms:

        name_algo = algo.name
        parameters = algo_param[name_algo]
        param_grid = ParameterGrid(parameters)
        alphas = rng.randint(1, 100, size=5) / 100
        alphas = np.sort(alphas)
        clf_est, offsets_est = est_tuning(X_train, X_test, algo, param_grid,
                                          alphas, U, vol_tot_cube)

        # check that clf_est gives the minimum auc
        score_function = clf_est.score_samples
        vol_est, _ = _compute_volumes(score_function, alphas,
                                      X_test, U, vol_tot_cube)
        auc_est = auc(alphas, vol_est)

        auc_algo = np.zeros(len(param_grid))
        for p, param in enumerate(param_grid):
            clf = algo(**param)
            clf = clf.fit(X_train)
            score_function_p = clf.score_samples
            vol_p, _ = _compute_volumes(score_function_p, alphas,
                                        X_test, U, vol_tot_cube)
            auc_algo[p] = auc(alphas, vol_p)

        assert_equal(np.min(auc_algo), auc_est)
        # TODO test on offsets_est when quantile instead of bisect


def test_anomaly_tuning():
    """Check anomaly_tuning gives same results than est_tuning."""

    parameters = {'k': np.arange(1, 10), 'novelty': [True]}
    alphas = np.array([0.2, 0.5, 0.9])
    models, offsets = anomaly_tuning(X, AverageKLPE, alphas=alphas,
                                     parameters=parameters,
                                     cv=cv, n_sim=100)
    score_estimators = np.empty((n_estimator, len(X)))
    for i in range(n_estimator):
        score_estimators[i, :] = models[i].score_samples(X)

    score_estimators_seq = np.empty((n_estimator, len(X)))
    offsets_seq = np.empty((n_estimator, len(alphas)))
    param_grid = ParameterGrid(parameters)
    i = 0
    for train, test in cv.split(X):
        X_train = X[train]
        X_test = X[test]

        model, offset = est_tuning(X_train, X_test, AverageKLPE,
                                   param_grid, alphas, U, vol_tot_cube)
        score_estimators_seq[i, :] = model.score_samples(X)
        offsets_seq[i, :] = offset
        i += 1

    assert_array_almost_equal(score_estimators, score_estimators_seq)
    assert_array_almost_equal(offsets, offsets_seq)
