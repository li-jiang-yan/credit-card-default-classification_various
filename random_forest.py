"""Random Forest
"""

from __future__ import annotations
from sklearn.metrics import classification_report
from random_forest_feature_selection import read_data, get_clf

# Functions for scalability
def classification_report_table(clf, x, y_true, labels, title):
    """
    Return a table of the classification report
    """
    y_pred = clf.predict(x)
    clf_report = classification_report(y_true, y_pred, labels=labels)
    return f"{title}\n{clf_report}"

# Select best column indexes based on randforest_feature_scores.txt
col_idxs = [0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

# Load data from pkl files
x_train = read_data(filepath="x_train.pkl", istarget=False)
y_train = read_data(filepath="y_train.pkl", istarget=True)
x_test = read_data(filepath="x_test.pkl", istarget=False)
y_test = read_data(filepath="y_test.pkl", istarget=True)

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
