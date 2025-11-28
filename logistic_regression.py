"""Logistic Regression
"""

from __future__ import annotations
from itertools import combinations, chain
from math import log
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from scipy.special import expit

# Functions for scalability
def read_data(filepath, istarget):
    """
    Return the feature/target data from a pkl file
    """
    df = pd.read_pickle(filepath)

    # If the data is a target DataFrame, apply ravel to suppress warnings from classifier
    if istarget:
        return np.ravel(df)

    return df

def get_clf():
    """
    Return a clean classifier with the parameters
    """
    return LogisticRegression(random_state=0, solver="liblinear", class_weight="balanced")

def score_feature_indices(x_tr, y_tr, x_te, y_te, cols):
    """
    Return the model testing accuracy for a given list of feature indices
    """
    # Make cols a tuple if it is not
    if not isinstance(cols, tuple):
        cols = tuple(cols)

    # Supress the warnings from the classifier
    y_tr = np.ravel(y_tr)
    y_te = np.ravel(y_te)

    # If we've previously scored and saved the score in the dictionary, we simply retrieve it
    # Otherwise, we score and save the score in the dictionary
    if cols in subset_scores:
        score = subset_scores[cols]
    else:
        clf = get_clf().fit(x_tr.iloc[:, list(cols)], y_tr)
        score = clf.score(x_te.iloc[:, list(cols)], y_te)
        subset_scores[cols] = score
        f.write(f"{str(cols):<82}: {score:.5f} \n")

    return score

def all_combinations(iterable):
    """
    Return all combinations of the iterable starting from length 1 to the length of the iterable
    """
    return chain.from_iterable(combinations(iterable, r) for r in range(1, len(iterable)+1))

def select_best_features(x_tr, y_tr, x_te, y_te, candidates):
    """
    Return the feature indices from a iterable of candidates that achieve the highest model
    accuracy
    """
    iterable = all_combinations(candidates)
    def key(cols):
        return score_feature_indices(x_tr, y_tr, x_te, y_te, cols)
    return max(iterable, key=key)

def select_best_features_groups(x_tr, y_tr, x_te, y_te, groups):
    """
    Return the groups of feature indices from a iterable of candidates that achieve the highest
    model accuracy
    """
    # Find the best feature indices from each group
    def function(group):
        return select_best_features(x_tr, y_tr, x_te, y_te, group)
    groups = list(map(function, groups))

    # Find the best groups of feature indices
    iterable = all_combinations(groups)
    def key(cols):
        return score_feature_indices(x_tr, y_tr, x_te, y_te, chain.from_iterable(cols))
    return list(chain.from_iterable(max(iterable, key=key)))

def classification_report_table(clf, x, y_true, labels, title):
    """
    Return a table of the classification report
    """
    y_pred = clf.predict(x)
    clf_report = classification_report(y_true, y_pred, labels=labels)
    return f"{title}\n{clf_report}"

def plot_logistic_regression(clf, x_te, y_te):
    """
    Plot the logistic regression curve
    """
    # Functions needed
    def logit(p):
        return log(p / (1-p))
    def inverse_logistic_curve(y):
        return (logit(y) - clf.intercept_) / clf.coef_
    def logistic_curve(x):
        return np.ravel(expit(x * clf.coef_ + clf.intercept_))
    def count_matches(x):
        matches = x_te.loc[y_te == legend] == x
        return matches.sum()
    def bar_height(x):
        bin_total = x_te == x
        return count_matches(x) / bin_total.sum()

    # Convert x to pd.Series if not
    if not isinstance(x_te, pd.Series):
        x_te = x_te.squeeze()

    # Plot logistic regression curve
    (x_min, x_max) = (inverse_logistic_curve(0.01), inverse_logistic_curve(0.99))
    x_curve = np.linspace(x_min, x_max).ravel()
    y_curve = logistic_curve(x_curve)
    plt.plot(x_curve, y_curve, color="red")

    # Plot bar charts
    x_bar = sorted(x_te.unique())
    bottom = np.zeros(len(x_bar))
    for legend in sorted(np.unique(y_te), reverse=True):
        bar_heights = list(map(bar_height, x_bar))
        bar_labels = list(map(count_matches, x_bar))
        p = plt.bar(x_bar, bar_heights, label=legend, bottom=bottom)
        plt.bar_label(p, labels=bar_labels, label_type="center")
        bottom += bar_heights

    plt.legend()
    plt.xlabel("Repayment status in Aug")
    plt.ylabel("Probability of default payment")
    plt.title("Probability of default payment vs. repayment status in Aug")

# Load data from pkl files
x_train = read_data(filepath="x_train.pkl", istarget=False)
y_train = read_data(filepath="y_train.pkl", istarget=True)
x_test = read_data(filepath="x_test.pkl", istarget=False)
y_test = read_data(filepath="y_test.pkl", istarget=True)

# Shortlist features
# If we do a naive brute force through all variables, we will have to deal with 2^23 combinations
# which will take too much time. On the other hand, we can group features in the various groups
# LIMIT_BAL                         (X1)
# SEX                               (X2)
# EDUCATION                         (X3)
# MARRIAGE                          (X4)
# AGE                               (X5)
# PAY_X: PAY_0 - PAY_6              (X6-X11)
# BILL_AMTX: BILL_AMT1 - BILL_AMT6  (X12-X17)
# PAY_AMTX: PAY_AMT1 - PAY_AMT6     (X18-X23)
# Less is more!

# Dictionary to store the scores for each subset
subset_scores = {}

# The various groups of column indices
col_groups = [[0], [1], [2], [3], [4], range(5,11), range(11,17), range(17,23)]

# Wipe contents of output file
with open("logreg_feature_scores.txt", "w", encoding="utf-8") as f:
    pass

# Open output file to read into
# and select the best column indexes to be used
with open("logreg_feature_scores.txt", "a", encoding="utf-8") as f:
    col_idxs = select_best_features_groups(x_train, y_train, x_test, y_test, col_groups) # [6]

# Train ML model
x_train = x_train.iloc[:, col_idxs]
x_test = x_test.iloc[:, col_idxs]
cls = get_clf()
cls.fit(x_train, y_train)

# Evaluate ML model training and testing performance
print(classification_report_table(cls, x_train, y_train, [0,1], \
                                  "classification_report(x_train, y_train)"))
print(classification_report_table(cls, x_test, y_test, [0,1], \
                                  "classification_report(x_test, y_test)"))

# Plot logistic regression model (only 1 feature)
plot_logistic_regression(cls, x_test, y_test)
plt.show()
