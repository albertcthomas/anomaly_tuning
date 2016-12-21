# Authors: Albert Thomas
#          Alexandre Gramfort
# License: BSD (3-clause)

import logging

import numpy as np

from scipy.optimize import bisect

from sklearn.model_selection import ParameterGrid
from sklearn.metrics import auc
from sklearn.utils import check_random_state
from sklearn.externals.joblib import Parallel, delayed

from .estimators import AverageKLPE

# Logging configuration
logging.basicConfig(level=logging.DEBUG,
                    format='%(levelname)s: %(message)s')


def est_tuning(X_train, X_test, base_estimator, param_grid,
               alphas, U, vol, vol_tot_cube):
    """Learn the best hyperparameters of base_estimator from the random
    splitting of the data set into X_train and X_test.

    Returns base_estimator instantiated with the best hyperparameter and
    fitted on X_train and the offsets of the score method corresponding to
    the probabilities of the alphas array.

    Parameters
    ----------
    X_train : array, shape (n_train, n_features)
        Training data set.

    X_test : array, shape (n_test, n_features)
        Test data set.

    base_estimator : object
        Anomaly detection estimator. Should have a fit method and a score
        method such that the smaller the score, the more abnormal the
        observation.

    param_grid : object
        Grid of hyperparameters from which the best hyperparameters should be
        selected.

    alphas : array, shape (n_alphas,)
        Probabilities of the Minimum Volume sets to estimate.

    U : array, shape (n_sim, n_features)
        Uniformly distributed samples to compute the volume of the estimated
        sets.

    vol : array, shape (n_alphas, n_hyperparameters)
        Indicator function of the estimated set of the samples U.

    vol_tot_cube : float
        Volume of the hypercube enclosing the data.

    Returns
    -------
    clf_est : base_estimator object
        base_estimator instantiated with the best hyperparameter and fitted on
        X_train.

    offsets_est : array, shape (n_alphas)
        Offsets of the score obtained from clf_est corresponding to the
        probabilities of the alphas array

    """

    auc_est = np.zeros(len(param_grid))

    for p, param in enumerate(param_grid):

        clf = base_estimator(**param)
        clf = clf.fit(X_train)
        clf_U = clf.score_samples(U)
        clf_test = clf.score_samples(X_test)
        min_test = np.min(clf_test)
        max_test = np.max(clf_test)

        for a, alpha in enumerate(alphas):

            def f(x):
                return np.mean((clf_test - x) >= 0) - alpha

            b_0 = bisect(f, min_test - 1e-12, max_test + 1e-12, xtol=1e-12,
                         maxiter=1000, full_output=False)
            b_0 = b_0 - 1e-12  # for robustness of the code

            U_inliers = (clf_U - b_0 >= 0)

            vol[a, p] = vol_tot_cube * np.mean(U_inliers)

        auc_est[p] = auc(alphas, vol[:, p])

    best_p = np.argmin(auc_est)

    best_param = param_grid[best_p]

    # Retraining the model with the best parameters
    clf_est = base_estimator(**best_param)
    clf_est.fit(X_train)

    clf_test = clf_est.score_samples(X_test)
    min_test = np.min(clf_test)
    max_test = np.max(clf_test)

    offsets_est = np.zeros(len(alphas))
    for a, alpha in enumerate(alphas):
        def f(x):
            return np.mean((clf_test - x) >= 0) - alpha

        b = bisect(f, min_test - 1e-12, max_test + 1e-12, xtol=1e-12,
                   maxiter=1000, full_output=False)
        b = b - 1e-12  # for robustness of the code

        offsets_est[a] = b

    return clf_est, offsets_est


def anomaly_tuning(X,
                   base_estimator=AverageKLPE,
                   parameters={'k': np.array([10])},
                   cv=None,
                   n_sim=10000,
                   alphas=np.arange(0.05, 1., 0.05),
                   random_state=42,
                   n_jobs=1,
                   ):
    """The data set X is randomly split into a training set X_train and a test
    set X_test. Given an anomaly detection algorithm, a scoring function is
    learnt on X_train. The best hyperparameter is selected using the area
    under the Mass Volume curve computed on X_test.

    Returns an ensemble of scoring functions built from several random
    splits and the offsets of each scoring function of the ensemble
    corresponding to the probabilities of alphas.

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        Data set.

    base_estimator : object , optional (default=AverageKLPE)
        Anomaly detection estimator. Should have a fit method and a score
        method such that the smaller the score, the more abnormal the
        observation. The possible anomaly detection algorithm are the ones
        defined in algos_subclasses.py.

    parameters : dict, optional (default={'k': np.array([10])})
        Grid of hyperparameters from which the best hyperparameters should be
        selected.

    cv : cross-validation generator
        Determines the cross-validation splitting strategy.

    n_sim : integer, optional (default=10000)
        Number of uniformly generated samples to compute the volumes of the
        estimated sets.

    alphas : array, shape (n_alphas,), optional (default=np.arange(0.05, 1.,
                                                                   0.05))
        Probabilities of the Minimum Volume sets to estimate.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for cross validation.
        If -1, then the number of jobs is set to the number of cores.

    Returns
    -------
    models : list, shape (n_estimator)
        Ensemble of anomaly detection estimators, each of them instantiated
        with the best hyperparameters learnt from their corresponding random
        split and fitted on X_train.

    offsets : array, shape (n_estimator, n_alphas)
        Offsets of the anomaly detection estimators of the ensemble
        corresponding to the probabilities of the alphas array

    """

    n_samples, n_features = X.shape
    if n_features >= 5:
        logging.warning(
            'Dimension might be too high for volume estimation '
            'and thus deteriorate model selection.',
        )

    param_grid = ParameterGrid(parameters)

    X_range = np.zeros((n_features, 2))
    X_range[:, 0] = np.min(X, axis=0)
    X_range[:, 1] = np.max(X, axis=0)

    # Hypercube sampling: sampling uniformly in X_range
    # Volume of the hypercube enclosing the data
    vol_tot_cube = np.prod(X_range[:, 1] - X_range[:, 0])
    rng = check_random_state(random_state)
    U = np.zeros((n_sim, n_features))
    for l in range(n_features):
        U[:, l] = rng.uniform(X_range[l, 0], X_range[l, 1], n_sim)

    vol = np.zeros((len(alphas), len(param_grid)))

    res = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(est_tuning)(
            X[train], X[test],
            base_estimator,
            param_grid,
            alphas,
            U,
            vol,
            vol_tot_cube)
        for train, test in cv.split(X))

    models = list(list(zip(*res))[0])
    offsets = np.array(list(zip(*res))[1])

    return models, offsets
