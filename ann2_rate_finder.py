"""Artificial neural networks learning rate finder (implementation with features 1, 2, 3, 4, 5,
 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22)
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score
import keras
from common_functions import read_data

# Constants
EPOCHS = 30
(RATE_MIN, RATE_MAX) = (1e-9, 0.014)

# Functions for scalability
def get_clf(ncols, lr):
    """
    Return a clean classifier with the parameters
    """
    result = keras.models.Sequential([
        keras.layers.Input(shape=(ncols,)),
        keras.layers.Dense(128, activation="relu", kernel_initializer="random_normal"),
        keras.layers.Dense(64, activation="relu", kernel_initializer="random_normal"),
        keras.layers.Dense(32, activation="relu", kernel_initializer="random_normal"),
        keras.layers.Dense(16, activation="relu", kernel_initializer="random_normal"),
        keras.layers.Dense(8, activation="relu", kernel_initializer="random_normal"),
        keras.layers.Dense(1, activation="sigmoid")
    ])

    optimizer = keras.optimizers.Adam(learning_rate=lr)

    result.compile(loss="binary_crossentropy", optimizer=optimizer)

    return result

# Process data columns
# Col | Info
#   1 | X2: Gender (1 = male; 2 = female).
#   2 | X3: Education (1 = graduate school; 2 = university; 3 = high school; 4 = others).
#   3 | X4: Marital status (1 = married; 2 = single; 3 = others).
#   4 | X5: Age (year).
#   5 | X6 = the repayment status in September, 2005;
#   6 | X7 = the repayment status in August, 2005;
#   7 | X8 = repayment status in Jul 2005
#   8 | X9 = repayment status in Jun 2005
#   9 | X10 = repayment status in May 2005
#  10 | X11 = the repayment status in April, 2005.
#  11 | X12 = amount of bill statement in September, 2005;
#  12 | X13 = amount of bill statement in August, 2005
#  13 | X14 = bill statement amount in Jul 2005
#  14 | X15 = bill statement amount in Jun 2005
#  15 | X16 = bill statement amount in May 2005
#  16 | X17 = amount of bill statement in April, 2005.
#  17 | X18 = amount paid in September, 2005;
#  18 | X19 = amount paid in August, 2005;
#  19 | X20 = amount paid in Jul 2005
#  20 | X21 = amount paid in Jun 2005
#  21 | X22 = amount paid in May 2005
#  22 | X23 = amount paid in April, 2005.

# One hot encode categorical columns
def concat_pd_objs(pd_objs):
    """Concatenate the rows of the given Series/DataFrames"""
    concat_layer = keras.layers.Concatenate(axis=1)
    return concat_layer(pd_objs)

def encode_df(df):
    """
    Encode the given DataFrame with one hot encoding
    """
    def encode_series(s):
        """Encode the given Series with one hot encoding"""
        encoding_layer = keras.layers.IntegerLookup(output_mode="one_hot")
        encoding_layer.adapt(s)
        return encoding_layer(s)

    # DataFrame -> Iterable of Series
    s_list = map(lambda i: df[i], df.columns)

    # Encode each Series
    s_list = list(map(encode_series, s_list))

    # Iterable of Series -> DataFrame
    return concat_pd_objs(s_list)

# Normalize numerical columns
def normalize_df(df):
    """
    Normalize the given DataFrame
    """
    a = df.to_numpy()
    normalization_layer = keras.layers.Normalization()
    normalization_layer.adapt(a)
    return normalization_layer(a)

def process_features(x):
    """
    Process the given features array
    """
    x_cat = x.iloc[:, 1:4]
    x_num = x.iloc[:, 4:]
    return concat_pd_objs([encode_df(x_cat), normalize_df(x_num)])

if __name__ == "__main__":

    def learning_rate_accuracy(lr):
        """Get the model accuracy for a given learning rate"""
        # Neural network training takes too long, so we just use the optimal columns 1, 2, 3, 4, 5,
        # 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22 found in other models
        ncolumns = x_tr.shape[1]

        # Get model
        model = get_clf(ncolumns, lr)

        # Fit model
        model.fit(x_tr, y_train, epochs=EPOCHS, class_weight=class_weight)

        # Get accuracy
        y_pred = model.predict(x_te) > 0.5
        return accuracy_score(y_test, y_pred)

    # Load data from pkl files
    x_train = read_data(filepath="x_train.pkl", istarget=False)
    y_train = read_data(filepath="y_train.pkl", istarget=True)
    x_test = read_data(filepath="x_test.pkl", istarget=False)
    y_test = read_data(filepath="y_test.pkl", istarget=True)

    # Process features
    x_tr = process_features(x_train)
    x_te = process_features(x_test)

    # Calculate class weights
    class_weight = dict(enumerate(compute_class_weight(class_weight="balanced",
                                                       classes=np.unique(y_train), y=y_train)))

    # Initialize horizontal and vertical axes of plot
    learning_rates = np.linspace(RATE_MIN, RATE_MAX, 30)
    accuracies = list(map(learning_rate_accuracy, learning_rates))

    # Plot accuracies vs learning rates
    plt.plot(learning_rates, accuracies, "-x")
    plt.xlabel("Learning rate")
    plt.ylabel("Accuracy")
    plt.title(f"Model accuracy after {EPOCHS} epochs vs learning rate")
    plt.show()
