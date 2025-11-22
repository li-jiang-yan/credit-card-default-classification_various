"""Data Preprocessing
"""

from __future__ import annotations
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split

# fetch dataset
default_of_credit_card_clients = fetch_ucirepo(id=350)

# data (as pandas dataframes)
X = default_of_credit_card_clients.data.features
y = default_of_credit_card_clients.data.targets

# split dataframes into train and test subsets
# the naming style is consistent with the sklearn documentation, but it doesn't conform to
# UPPER_CASE naming style per pylint
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
