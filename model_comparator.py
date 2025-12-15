"""Compares all the models developed
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from common_functions import read_data
import keras
import logreg_feature_selection as logreg
import randforest_feature_selection as randforest
import svm_feature_selection as svm
import ann1_rate_finder as ann1
import ann2_rate_finder as ann2

# Constants
PADDING = -13

# Load data from pkl files
x_train = read_data(filepath="x_train.pkl", istarget=False)
y_train = read_data(filepath="y_train.pkl", istarget=True)
x_test = read_data(filepath="x_test.pkl", istarget=False)
y_test = read_data(filepath="y_test.pkl", istarget=True)

# Models
models = ["LR", "RF-1F", "RF-MF", "SVM-L", "ANN-1F", "ANN-MF"]
labels = ["0", "1"]
metrics = ["precision", "recall", "f1-score", "accuracy"]
cls_report = {
    "0": {"precision": [], "recall": [], "f1-score": [], "support": []},
    "1": {"precision": [], "recall": [], "f1-score": [], "support": []},
    "accuracy": [],
    "macro avg": {"precision": [], "recall": [], "f1-score": [], "support": []},
    "weighted avg": {"precision": [], "recall": [], "f1-score": [], "support": []}
}

# Function to append classification reports
def append_cls_report(y_te, y_pr):
    """Generates a classification report and appends it to cls_report"""
    clf_report = classification_report(y_te, y_pr, output_dict=True)
    for k0, v0 in cls_report.items():
        if isinstance(v0, list):
            v0.append(clf_report[k0])
        else:
            for k1, v1 in v0.items():
                v1.append(clf_report[k0][k1])

# Logistic regression
cls = logreg.get_clf().fit(x_train.iloc[:, [6]], y_train)
y_pred = cls.predict(x_test.iloc[:, [6]])
append_cls_report(y_test, y_pred)

# Random forest (only feature 6)
cls = randforest.get_clf().fit(x_train.iloc[:, [6]], y_train)
y_pred = cls.predict(x_test.iloc[:, [6]])
append_cls_report(y_test, y_pred)

# Random forest (features 1 and beyond)
cls = randforest.get_clf().fit(x_train.iloc[:, 1:], y_train)
y_pred = cls.predict(x_test.iloc[:, 1:])
append_cls_report(y_test, y_pred)

# Linear support vector machine
cls = svm.get_clf((6,)).fit(x_train.iloc[:, [6]], y_train)
y_pred = cls.predict(x_test.iloc[:, [6]])
append_cls_report(y_test, y_pred)

# Artificial neural networks (only feature 6)
keras.utils.set_random_seed(0)
class_weight = dict(enumerate(compute_class_weight(class_weight="balanced",
                                                    classes=np.unique(y_train), y=y_train)))
cls = ann1.get_clf((6,), lr=0.001)
cls.fit(x_train.iloc[:, [6]], y_train, epochs=100, class_weight=class_weight)
y_pred = np.ravel(cls.predict(x_test.iloc[:, [6]]) > 0.5)
append_cls_report(y_test, y_pred)

# Artificial neural networks (features 1 and beyond)
x_tr = ann2.process_features(x_train)
x_te = ann2.process_features(x_test)
cls = ann2.get_clf(x_tr.shape[1], lr=0.012)
cls.fit(x_tr, y_train, epochs=100, class_weight=class_weight)
y_pred = np.ravel(cls.predict(x_te) > 0.5)
append_cls_report(y_test, y_pred)

# Create subplots
fig, axs = plt.subplots(2, 2)

# Precision subplot
ax = axs[0, 0]
p0 = ax.bar(models, cls_report["0"]["precision"], color="deepskyblue")
p1 = ax.bar(models, cls_report["1"]["precision"], color="darkorange")
ax.bar_label(p0, fmt="%.3f", padding=PADDING)
ax.bar_label(p1, fmt="%.3f", padding=PADDING)
ax.set_title("Precision")

# Recall subplot
ax = axs[0, 1]
p0 = ax.bar(models, cls_report["0"]["recall"], color="deepskyblue")
p1 = ax.bar(models, cls_report["1"]["recall"], color="darkorange")
ax.bar_label(p0, fmt="%.3f", padding=PADDING)
ax.bar_label(p1, fmt="%.3f", padding=PADDING)
ax.set_title("Recall")

# F1-score subplot
ax = axs[1, 0]
p0 = ax.bar(models, cls_report["0"]["f1-score"], color="deepskyblue")
p1 = ax.bar(models, cls_report["1"]["f1-score"], color="darkorange")
ax.bar_label(p0, fmt="%.3f", padding=PADDING)
ax.bar_label(p1, fmt="%.3f", padding=PADDING)
ax.set_title("F1-score")

# Accuracy subplot
ax = axs[1, 1]
p = ax.bar(models, cls_report["accuracy"], color="deepskyblue")
ax.bar_label(p, fmt="%.3f", padding=PADDING)
ax.set_title("Accuracy")

# Show subplots
plt.show()
