"""Logistic Regression
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

col_idxs_list = [
    [6],
    [1, 6],
    [0, 1, 2, 4, 6, 11, 12, 13, 14, 15, 16],
    [0, 2, 4, 6, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
    [0, 1, 2, 4, 6, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
]

for col_idxs in col_idxs_list:
    print(f"col_idxs = {col_idxs}")
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
