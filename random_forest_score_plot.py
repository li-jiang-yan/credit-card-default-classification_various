"""Plots for Random Forest
"""
import re
import matplotlib.pyplot as plt

# Read records from randforest_feature_scores.txt
records = []

def get_record(m):
    """Return a tuple of (features, train, test) from a line re.Match object"""
    features = m.group(1)
    train = float(m.group(2))
    test = float(m.group(3))
    return (features, train, test)

with open("randforest_feature_scores.txt", "r", encoding="utf-8") as f:
    for line in f:
        match = re.search(r"(\([\d,\s]+\))\s+(0\.\d{5})\s(0\.\d{5})", line)
        if match:
            record = get_record(match)
            records.append(record)

# Plot each record in matplotlib
def get_train(rec):
    """Return training accuracy from record"""
    return rec[1]

def get_test(rec):
    """Return test accuracy from record"""
    return rec[2]

def is_good(rec):
    """Return whether record is good (training accuracy > 0.95 & testing accuracy > 0.80)"""
    train = get_train(rec)
    test = get_test(rec)
    return train > 0.9 and test > 0.8175

x = [get_train(record) for record in records if is_good(record)]
y = [get_test(record) for record in records if is_good(record)]

plt.plot(x, y, '.')
plt.xlabel("Training accuracy")
plt.ylabel("Testing accuracy")
plt.title("Random forest testing accuracy vs training accuracy")
plt.show()
