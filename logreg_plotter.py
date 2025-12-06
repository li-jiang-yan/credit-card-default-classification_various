"""Plots for logistic regression
"""

from __future__ import annotations
from math import log
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
from common_functions import read_data
from logreg_feature_selection import get_clf

def train(x_tr, y_tr):
    """
    Train the logistic regression model
    """
    x_arr = x_tr.iloc[:, [6]]
    clf = get_clf()
    clf.fit(x_arr, y_tr)
    return clf

def plot(clf, x_te, y_te):
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
        matches = x_arr.loc[y_te == legend] == x
        return matches.sum()
    def bar_height(x):
        bin_total = x_arr == x
        return count_matches(x) / bin_total.sum()

    # Convert x to pd.Series of 1 column
    x_arr = x_te.iloc[:, 6].squeeze()

    # Plot logistic regression curve
    (x_min, x_max) = (inverse_logistic_curve(0.01), inverse_logistic_curve(0.99))
    x_curve = np.linspace(x_min, x_max).ravel()
    y_curve = logistic_curve(x_curve)
    plt.plot(x_curve, y_curve, color="red")

    # Plot bar charts
    x_bar = sorted(x_arr.unique())
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
    plt.show()

# Load data from pkl files
x_train = read_data(filepath="x_train.pkl", istarget=False)
y_train = read_data(filepath="y_train.pkl", istarget=True)
x_test = read_data(filepath="x_test.pkl", istarget=False)
y_test = read_data(filepath="y_test.pkl", istarget=True)

# Train and plot model
cls = train(x_train, y_train)
plot(cls, x_test, y_test)
