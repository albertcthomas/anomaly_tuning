import numpy as np

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_true
from sklearn.utils.testing import assert_array_equal

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

X_range = np.zeros((n_features, 2))
X_range[:, 0] = np.min(X, axis=0)
X_range[:, 1] = np.max(X, axis=0)

# Hypercube sampling: sampling uniformly in X_range
# Volume of the hypercube enclosing the data
vol_tot_cube = np.prod(X_range[:, 1] - X_range[:, 0])
n_sim = 100
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

    alphas = rng.randint(10, size=5) / 10
    vol, offset = _compute_volumes(score_function_toy, alphas, grid, grid, 1.)
    assert_array_equal(alphas, vol)

    # TODO add test for offset when bisect is replaced by quantile


def test_compute_volumes():
    """Check _compute_volumes for several masses."""
    estimators = [AverageKLPE(k=3, novelty=True), MaxKLPE(k=3, novelty=True),
                  OCSVM(sigma=1.),
                  IsolationForest(n_estimators=10, random_state=2),
                  KernelSmoothing()]
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=23)
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
        # check increasing order of volumes and descreasing order of offset
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


def test_anomaly_tuning():

    expected_offsets = {'aklpe': np.array([[-0.27114216, -0.33649402],
                                           [-0.60454928, -0.61773153]]),
                        'mklpe': np.array([[-0.96130544, -1.17929608],
                                           [-0.70046052, -0.71327533]]),
                        'ocsvm': np.array([[0.05338662, -0.03524815],
                                           [0.20192307, -0.23859429]]),
                        'iforest': np.array([[0.01947895, 0.00902796],
                                             [0.03771182, -0.00537859]]),
                        'ks': np.array([[-3.07824835, -3.19763788],
                                        [-3.08133308, -3.18917259]])
                        }

    for algo in algorithms:
        name_algo = algo.name
        parameters = algo_param[name_algo]
        _, offsets = anomaly_tuning(X, base_estimator=algo,
                                    alphas=np.arange(0.7, 0.8, 0.1),
                                    parameters=parameters,
                                    random_state=42, cv=cv)
        assert_array_almost_equal(expected_offsets[name_algo], offsets)
