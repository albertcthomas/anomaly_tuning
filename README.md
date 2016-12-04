# Learning hyperparameters for unsupervised anomaly detection

This repository contains the Python code to learn hyperparameters of unsupervised anomaly detection algorithms as described in the paper ["Learning hyperparameters for unsupervised anomaly detection", A. Thomas, S. Clémençon, V. Feuillard, A. Gramfort, Anomaly Detection Workshop, ICML 2016.](https://drive.google.com/file/d/0B8Dg3PBX90KNUTg5NGNOVnFPX0hDNmJsSTcybzZMSHNPYkd3/view)

Description of the files:

* estimators.py : anomaly detection estimators used in the paper.
* tuning.py : hyperparameters selection algorithm described in the paper.
* examples.py : example of the algorithm on a two dimensional Gaussian mixture data set.
* utils.py : Gaussian mixture object used to sample from a Gaussian mixture model and compute the density of the model.

Requirements:

* Python (version 2.7)
* [scikit-learn](http://scikit-learn.org/stable/) (at least version 0.18)

