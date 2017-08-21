import numpy as np

from sklearn import ensemble
from sklearn.svm import OneClassSVM
from sklearn.utils.testing import assert_array_equal

from anomaly_tuning.estimators import OCSVM
from anomaly_tuning.estimators import KLPE
from anomaly_tuning.estimators import AverageKLPE
from anomaly_tuning.estimators import MaxKLPE
from anomaly_tuning.estimators import IsolationForest
from anomaly_tuning.estimators import KernelSmoothing


def test_score_samples_estimators():
    """Check score_samples derived from sklearn decision_function.

    Check that the values are the same. This only concerns OCSVM and
    IsolationForest."""

    X = np.random.randn(50, 2)

    clf_1 = IsolationForest(random_state=88)
    clf_1.fit(X)

    clf_2 = ensemble.IsolationForest(random_state=88)
    clf_2.fit(X)

    assert_array_equal(clf_1.score_samples(X), clf_2.decision_function(X))

    nu = 0.4
    sigma = 3.0
    gamma = gamma = 1. / (2. * sigma ** 2)
    clf_1 = OCSVM(sigma=sigma, nu=nu)
    clf_1.fit(X)

    clf_2 = OneClassSVM(gamma=gamma, nu=nu)
    clf_2.fit(X)

    assert_array_equal(clf_1.score_samples(X),
                       clf_2.decision_function(X).ravel())
