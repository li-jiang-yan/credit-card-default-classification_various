"""Data Preprocessing
"""

from __future__ import annotations
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split

# fetch dataset
default_of_credit_card_clients = fetch_ucirepo(id=350)

# data (as pandas dataframes)
x = default_of_credit_card_clients.data.features
y = default_of_credit_card_clients.data.targets

# split dataframes into train and test subsets
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

# save variables in pkl files
x_train.to_pickle("x_train.pkl")
y_train.to_pickle("y_train.pkl")
x_test.to_pickle("x_test.pkl")
y_test.to_pickle("y_test.pkl")
