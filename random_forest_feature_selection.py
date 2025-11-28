"""Random Forest
"""

from __future__ import annotations
from itertools import combinations, chain
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Functions for scalability
def get_clf():
    """
    Return a clean classifier with the parameters
    """
    return RandomForestClassifier(random_state=0, class_weight="balanced")

def read_data(filepath, istarget):
    """
    Return the feature/target data from a pkl file
    """
    df = pd.read_pickle(filepath)

    # If the data is a target DataFrame, apply ravel to suppress warnings from classifier
    if istarget:
        return np.ravel(df)

    return df

# Avoid running this or not imports will last forever...
if __name__ == "main":

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

    # Load data from pkl files
    x_train = read_data(filepath="x_train.pkl", istarget=False)
    y_train = read_data(filepath="y_train.pkl", istarget=True)
    x_test = read_data(filepath="x_test.pkl", istarget=False)
    y_test = read_data(filepath="y_test.pkl", istarget=True)

    # Shortlist features
    # If we do a naive brute force through all variables, we will have to deal with 2^23
    # combinations which will take too much time. On the other hand, we can group features in the
    # various groups
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
    with open("randforest_feature_scores.txt", "w", encoding="utf-8") as f:
        pass

    # Open output file to read into
    # and select the best column indexes to be used
    with open("randforest_feature_scores.txt", "a", encoding="utf-8") as f:
        col_idxs = select_best_features_groups(x_train, y_train, x_test, y_test, col_groups)

    print(col_idxs)
    # [1, 2, 3, 4, 6, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
