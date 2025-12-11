"""Artificial neural networks learning rate finder (implementation with only feature 6)
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
(RATE_MIN, RATE_MAX) = (1e-9, 0.0018)

# Functions for scalability
def get_clf(cols, lr):
    """
    Return a clean classifier with the parameters
    """
    result = keras.models.Sequential([
        keras.layers.Input(shape=(len(cols),)),
        keras.layers.Normalization(),
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

if __name__ == "__main__":

    def learning_rate_accuracy(lr):
        """Get the model accuracy for a given learning rate"""
        # Neural network training takes too long, so we just use the optimal column 6 found in other
        # models
        columns = (6,)

        # Get model
        model = get_clf(columns, lr)

        # Fit model
        model.fit(x_train.iloc[:, list(columns)], y_train, epochs=EPOCHS, class_weight=class_weight)

        # Get accuracy
        y_pred = model.predict(x_test.iloc[:, list(columns)]) > 0.5
        return accuracy_score(y_test, y_pred)

    # Load data from pkl files
    x_train = read_data(filepath="x_train.pkl", istarget=False)
    y_train = read_data(filepath="y_train.pkl", istarget=True)
    x_test = read_data(filepath="x_test.pkl", istarget=False)
    y_test = read_data(filepath="y_test.pkl", istarget=True)

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
