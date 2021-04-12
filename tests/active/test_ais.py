from unittest import TestCase

import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

from robustnessgym.active import ais


# Get labels and probabilities from trained sklearn model
def get_labels_and_probs(X_test, clf, rank_prob=False):
    if hasattr(clf, "predict_proba"):
        prob_pos = clf.predict_proba(X_test)[:, 1]
    else:  # use decision function
        prob_pos = clf.decision_function(X_test)
        prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
    if rank_prob:
        temp = prob_pos.argsort()
        ranks = np.empty_like(temp).astype(float)
        ranks[temp] = np.linspace(0.0, 0.99999, len(prob_pos))
        prob_pos = ranks
    y_pred = clf.predict(X_test)
    return y_pred, prob_pos


# Generate synthetic data to test model validation
def make_data(n_samples=100000, weights=(0.9, 0.1)):
    X, y = datasets.make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=2,
        n_redundant=2,
        weights=weights,
    )

    pos_ind = np.where(y == 1)[0]
    neg_ind = np.where(y == 0)[0]
    train_samples = 100  # Samples used for training the models
    # Randomly choose train_samples positive and negative rows
    pos_samples = np.random.choice(len(pos_ind), size=train_samples, replace=False)
    neg_samples = np.random.choice(len(neg_ind), size=train_samples, replace=False)
    train_indices = np.concatenate([pos_ind[pos_samples], neg_ind[neg_samples]])

    X_train = X[train_indices, :]
    X_test = np.delete(X, train_indices, axis=0)
    y_train = y[train_indices]
    y_test = np.delete(y, train_indices)
    return X_train, X_test, y_train, y_test


# Train a model on synthetic data, extract labels and probabilities
# from the trained model, then run AIS to validate
class TestAIS(TestCase):
    def test_endtoend(self):
        X_train, X_test, y_train, y_test = make_data(weights=(0.99, 0.01))
        clf = LogisticRegression()
        clf.fit(X_train, y_train)
        y_pred, probs = get_labels_and_probs(X_test, clf, rank_prob=True)
        prf1, stds, budget = ais.ais_fullalgorithm(y_pred, y_test, probs, 6000)
        self.assertTrue(budget <= 6000)
        self.assertTrue(np.nansum(prf1 > 1) + np.nansum(prf1 < 0) == 0)
