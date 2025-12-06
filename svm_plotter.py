"""Plots for support vector machine
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from svm_feature_selection import get_clf, read_data

def train(x_tr, y_tr):
    """
    Train the linear SVM model
    """
    cols = [6]
    x_arr = x_tr.iloc[:, cols]
    clf = get_clf(cols)
    clf.fit(x_arr, y_tr)
    return clf

def get_confusion_matrix(clf, x_te, y_te):
    """
    Get confusion matrix for classifier
    """
    y_pred = clf.predict(x_te)
    return confusion_matrix(y_te, y_pred)

def plot(x_te, y_te):
    """
    Plot the SVM curve
    """
    # Functions needed
    def count_matches(x):
        matches = x_arr.loc[y_te == legend] == x
        return matches.sum()
    def bar_height(x):
        bin_total = x_arr == x
        return count_matches(x) / bin_total.sum()

    # Convert x to pd.Series of 1 column
    x_arr = x_te.iloc[:, 6].squeeze()

    # Plot hyperplane
    # From confusion matrix, the hyperplane is x = 0.5
    y_sv = [0, 1]
    x_sv = [0.5] * len(y_sv)
    plt.plot(x_sv, y_sv, "k", label="hyperplane")

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

# Train and test SVC model
cls = train(x_train, y_train)
print("Confusion matrix:\n", get_confusion_matrix(cls, x_test, y_test))

# Plot SVC model
plot(x_test, y_test)
