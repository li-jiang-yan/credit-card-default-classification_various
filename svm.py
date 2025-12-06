"""Support Vector Machine
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from svm_feature_selection import get_clf, read_data

# Load data from pkl files
x_train = read_data(filepath="x_train.pkl", istarget=False)
y_train = read_data(filepath="y_train.pkl", istarget=True)
x_test = read_data(filepath="x_test.pkl", istarget=False)
y_test = read_data(filepath="y_test.pkl", istarget=True)

# Train and test SVC model
cols = [6]
clf = get_clf(cols).fit(x_train.iloc[:, cols], y_train)
y_pred = clf.predict(x_test)
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

def plot_svm(x_te, y_te):
    """
    Plot the logistic regression curve
    """
    # Functions needed
    def count_matches(x):
        matches = x_te.loc[y_te == legend] == x
        return matches.sum()
    def bar_height(x):
        bin_total = x_te == x
        return count_matches(x) / bin_total.sum()

    # Convert x to pd.Series if not
    if not isinstance(x_te, pd.Series):
        x_te = x_te.squeeze()

    # Plot hyperplane
    # From confusion matrix, the hyperplane is x = 0.5
    y_sv = [0, 1]
    x_sv = [0.5] * len(y_sv)
    plt.plot(x_sv, y_sv, "k", label="hyperplane")

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

plot_svm(x_test.iloc[:, 6], y_test)
plt.show()
