# Learning hyperparameters for unsupervised anomaly detection

This repository contains the Python code to learn hyperparameters of unsupervised anomaly detection algorithms as described in the paper ["Learning hyperparameters for unsupervised anomaly detection", A. Thomas, S. Clémençon, V. Feuillard, A. Gramfort, Anomaly Detection Workshop, ICML 2016.](https://drive.google.com/file/d/0B8Dg3PBX90KNUTg5NGNOVnFPX0hDNmJsSTcybzZMSHNPYkd3/view)

To install the package, run:

	python setup.py install --user

A demo is presented in the notebook: demo_automaly_tuning.ipynb

Description of the files:

* estimators.py : anomaly detection estimators used in the paper.
* tuning.py : hyperparameters selection algorithm described in the paper.
* examples.py : example of the algorithm on a two dimensional Gaussian mixture data set.
* utils.py : Gaussian mixture object used to sample from a Gaussian mixture model and compute the density of the model.

We are actively trying to reduce the number of dependencies. However, as of
now these are the dependencies for the examples to run:

* numpy (>=1.8)
* matplotlib (>=1.3)
* scipy (>=0.16)
* scikit-learn (0.18)

Code should run on both Python 2.7 and Python 3.4 or higher.
