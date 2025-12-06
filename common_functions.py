"""Commonly used functions
"""

from __future__ import annotations
from itertools import combinations, chain
import numpy as np
import pandas as pd

def read_data(filepath, istarget):
    """
    Return the feature/target data from a pkl file
    """
    df = pd.read_pickle(filepath)

    # If the data is a target DataFrame, apply ravel to suppress warnings from classifier
    if istarget:
        return np.ravel(df)

    return df

def all_combinations(iterable):
    """
    Return all combinations of the iterable starting from length 1 to the length of the iterable
    """
    return chain.from_iterable(combinations(iterable, r) for r in range(1, len(iterable)+1))
