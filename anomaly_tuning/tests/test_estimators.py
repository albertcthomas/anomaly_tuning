import numpy as np

from sklearn import ensemble
from sklearn.svm import OneClassSVM
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_array_almost_equal

from anomaly_tuning.estimators import OCSVM
from anomaly_tuning.estimators import KLPE
from anomaly_tuning.estimators import AverageKLPE
from anomaly_tuning.estimators import MaxKLPE
from anomaly_tuning.estimators import IsolationForest
from anomaly_tuning.estimators import KernelSmoothing


def test_score_samples_estimators():
    """Check that the values of score_samples methods derived from sklearn
    decision_functions are the same than sklearn decision_function methods.
    This only concerns OCSVM and IsolationForest."""

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


def test_averageklpe_score_samples():
    """Check score_samples for average KLPE"""

    X_train = np.array([[0, 0], [1, 1], [3, 1]])
    score_train_true = - np.array([(np.sqrt(2) + np.sqrt(10)) / 2,
                                   (np.sqrt(2) + 2) / 2,
                                   (2 + np.sqrt(10)) / 2])
    X_test = np.array([[-1, 0], [-1, 1]])
    score_test_true = - np.array([(1 + np.sqrt(5)) / 2,
                                  (np.sqrt(2) + 2) / 2])

    # check scores when novelty=False, i.e. scores of X_train itself
    clf = AverageKLPE(k=2)
    clf.fit(X_train)
    score_train = clf.score_samples(X_train)
    assert_array_almost_equal(score_train, score_train_true)
    # check scores when novelty=True, i.e. scores of test samples
    clf = AverageKLPE(k=2, novelty=True)
    clf.fit(X_train)
    score_test = clf.score_samples(X_test)
    assert_array_almost_equal(score_test, score_test_true)


def test_maxklpe_score_samples():
    """Check score_samples for max KLPE"""

    X_train = np.array([[0, 0], [1, 1], [3, 1]])
    score_train_true = - np.array([np.sqrt(10), 2, np.sqrt(10)])
    X_test = np.array([[-1, 0], [-1, 1]])
    score_test_true = - np.array([np.sqrt(5), 2])

    # check scores when novelty=False, i.e. scores of X_train itself
    clf = MaxKLPE(k=2)
    clf.fit(X_train)
    score_train = clf.score_samples(X_train)
    assert_array_almost_equal(score_train, score_train_true)
    # check scores when novelty=True, i.e. scores of test samples
    clf = MaxKLPE(k=2, novelty=True)
    clf.fit(X_train)
    score_test = clf.score_samples(X_test)
    assert_array_almost_equal(score_test, score_test_true)
