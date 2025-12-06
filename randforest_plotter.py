"""Plots for random forest
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

def plot():
    """Plot graphs for random forest"""
    def plot_ax(ax, function, title):
        """Plot subplot"""
        x = list(map(get_train, filter(function, records)))
        y = list(map(get_test, filter(function, records)))
        ax.plot(x, y, ".")
        ax.set_xlabel("Training accuracy")
        ax.set_ylabel("Testing accuracy")
        ax.set_title(title)
    def is_accurate(rec):
        """Return whether record is accurate"""
        return get_test(rec) > 0.80 and get_train(rec) > 0.995
    _, axs = plt.subplots(nrows=1, ncols=2, figsize=[12.8, 4.8])
    plot_ax(axs[0], lambda rec: True, "All points")
    plot_ax(axs[1], is_accurate, "Test accuracy > 0.80, train accuracy > 0.995")
    plt.suptitle("Testing accuracy vs training accuracy")
    plt.show()

# Plot
plot()
