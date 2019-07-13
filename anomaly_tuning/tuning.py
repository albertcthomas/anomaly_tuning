# Authors: Albert Thomas
#          Alexandre Gramfort
# License: BSD (3-clause)

import numpy as np

from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import auc
from sklearn.utils import check_random_state
from joblib import Parallel, delayed

from .estimators import AverageKLPE
from .tree import RegressionTree


def _compute_volumes(score_function, alphas, X_train, X_test,
                     volume_computation, U, vol_tot_cube):
    """Compute the volumes of each level set of the scoring function

    Each level set is associated to a mass of alphas. The function returns the
    volume and the offset of each level set.
    """
    score_test = score_function(X_test)

    # compute offsets
    offsets_p = np.percentile(score_test, 100 * (1 - alphas))
    # compute volumes of associated level sets
    if volume_computation == 'monte-carlo':
        score_U = score_function(U)
        vol_p = np.array([np.mean(score_U >= offset) for offset in offsets_p])
        vol_p = vol_p * vol_tot_cube
        tree_perf = None

    elif volume_computation == 'tree':
        # we fit the regression tree on the whole data set
        X = np.concatenate([X_train, X_test], axis=0)

        n_features = X.shape[1]
        X_range = np.zeros((n_features, 2))
        X_range[:, 0] = np.min(X, axis=0)
        X_range[:, 1] = np.max(X, axis=0)

        # regression tree grid search cv
        # min_sample_leaf grid has to be a list to preserve type: if numpy
        # array then 1 is cast as a float...
        tree_grid = {'min_samples_leaf': [0.01, 0.05, 0.1, 0.2, 0.3,
                                          0.4, 0.5, 1]}
        reg = GridSearchCV(RegressionTree(), tree_grid, cv=5)
        reg.fit(X, score_function(X))
        reg_best = reg.best_estimator_
        tree_perf = reg.best_score_
        vol_p = np.array([reg_best.volume_leafs(X_range, offset) for offset in
                          offsets_p])

    else:
        raise ValueError('Unknown volume computation method.')

    return vol_p, offsets_p, tree_perf


def _est_tuning(X_train, X_test, base_estimator, param_grid,
                alphas, volume_computation, U, vol_tot_cube, random=False,
                random_state=42):
    """Learn the best hyperparameters of base_estimator from the random
    splitting of the data set into X_train and X_test.

    Returns base_estimator instantiated with the best hyperparameter and
    fitted on X_train and the offsets of the score_sample method corresponding
    to the probabilities of the alphas array.

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

    volume_computation : string, optional (default='monte-carlo')
        The strategy used to compute the volume of the minimum volume set
        estimates. If 'monte-carlo' the volume is computed by generating
        samples uniformly in the hypercube enclosing the data. If 'tree' the
        scoring function is estimated with a regression tree. The volume is
        then approximated by the sum of the volume of the leafs characterizing
        the minimum volume set.

    U : array, shape (n_sim, n_features)
        Uniformly distributed samples to compute the volume of the estimated
        sets when volume_computation='monte-carlo'. Otherwise None.

    vol_tot_cube : float
        Volume of the hypercube enclosing the data when
        volume_computation='monte-carlo'. Otherwise None.

    random : bool, optional
        If true the parameter is randomly chosen in the grid.

    random_state : int, optional
        Seed used by the random number generator.

    Returns
    -------
    clf_est : base_estimator object
        base_estimator instantiated with the best hyperparameter and fitted on
        X_train.

    offsets_est : array, shape (n_alphas)
        Offsets of the score obtained from clf_est corresponding to the
        probabilities of the alphas array

    tree_perfs : array, shape (n_parameters,) or None
        Cross validated performance of the regression trees used to compute the
        volumes when volume_computation='tree'. Otherwise None.

    auc_est : array, shape (n_parameters,)
        Area under estimated MV curves for all parameters. Returned to plot
        cross validation curves.
    """
    offsets_all = np.zeros((len(alphas), len(param_grid)))
    auc_est = np.zeros(len(param_grid))
    if volume_computation == 'tree':
        tree_perfs = np.zeros(len(param_grid))
    else:
        tree_perfs = None

    if random:
        rng = np.random.RandomState(random_state)
        param = rng.choice(list(param_grid))
        clf_est = base_estimator(**param)
        clf_est = clf_est.fit(X_train)

        score_test = clf_est.score_samples(X_test)
        offsets = np.percentile(score_test, 100 * (1 - alphas))

        return clf_est, offsets, None, None

    # Grid search of best hyperparameters
    for p, param in enumerate(param_grid):
        # fit classifier with given parameters
        clf = base_estimator(**param)
        clf = clf.fit(X_train)
        score_function = clf.score_samples

        vol_p, offsets_p, tree_perf_p = _compute_volumes(
            score_function, alphas, X_train, X_test, volume_computation,
            U, vol_tot_cube)

        if volume_computation == 'tree':
            tree_perfs[p] = tree_perf_p

        auc_est[p] = auc(alphas, vol_p)
        offsets_all[:, p] = offsets_p

    best_p = np.argmin(auc_est)
    best_param = param_grid[best_p]

    # Retraining the model with the best parameters and recovering associated
    # offsets
    clf_est = base_estimator(**best_param)
    clf_est.fit(X_train)
    offsets_est = offsets_all[:, best_p]

    return clf_est, offsets_est, tree_perfs, auc_est


def anomaly_tuning(X,
                   base_estimator=AverageKLPE,
                   parameters={'k': np.array([10])},
                   cv=None,
                   alphas=np.arange(0.05, 1., 0.05),
                   volume_computation='monte-carlo',
                   n_sim=10000,
                   random_state=42,
                   n_jobs=1,
                   verbose=0,
                   random=False,
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

    alphas : array, shape (n_alphas,), optional (default=np.arange(0.05, 1.,
                                                                   0.05))
        Probabilities of the Minimum Volume sets to estimate.

    volume_computation : string, optional (default='monte-carlo')
        The strategy used to compute the volume of the minimum volume set
        estimates. If 'monte-carlo' the volume is computed by generating
        samples uniformly in the hypercube enclosing the data. If 'tree' the
        scoring function is estimated with a regression tree. The volume is
        then approximated by the sum of the volume of the leafs characterizing
        the minimum volume set.

    n_sim : integer, optional (default=10000)
        Number of uniformly generated samples to compute the volumes of the
        estimated sets when volume_computation='monte-carlo'.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    n_jobs : int, optional (default=1)
        The number of jobs to run in parallel for cross validation.
        If -1, then the number of jobs is set to the number of cores.

    verbose : int, optional (default=0)
        Controls the verbosity when fitting the estimator for each cv split.

    random : bool, optional (default=False)
        If true the parameter is randomly chosen in the grid.

    Returns
    -------
    models : list, shape (n_estimator)
        Ensemble of anomaly detection estimators, each of them instantiated
        with the best hyperparameters learnt from their corresponding random
        split and fitted on X_train.

    offsets : array, shape (n_estimator, n_alphas)
        Offsets of the anomaly detection estimators of the ensemble
        corresponding to the probabilities of the alphas array

    tree_perfs : array, shape (n_estimators, n_parameters) or None
        Cross validated performance of the regression trees used to compute the
        volumes when volume_computation='tree'. Otherwise None.

    auc_est : array, shape (n_estimators, n_parameters)
        Area under estimated MV curves for all parameters and all random
        splits. Returned to plot cross validation curves.
    """

    n_samples, n_features = X.shape

    param_grid = ParameterGrid(parameters)

    if volume_computation == 'monte-carlo':
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
    elif volume_computation == 'tree':
        U = None
        vol_tot_cube = None
    else:
        raise ValueError('Unknown volume computation method.')

    res = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_est_tuning)(
            X[train], X[test],
            base_estimator,
            param_grid,
            alphas,
            volume_computation,
            U,
            vol_tot_cube,
            random,
            random_state)
        for train, test in cv.split(X))

    models = list(list(zip(*res))[0])
    offsets = np.array(list(zip(*res))[1])

    if volume_computation == 'tree':
        tree_perfs = np.array(list(zip(*res))[2])
    else:
        tree_perfs = None

    auc_est = np.array(list(zip(*res))[3])

    return models, offsets, tree_perfs, auc_est
