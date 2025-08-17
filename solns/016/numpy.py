import numpy as np


def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
    # let's actually implement and then use the numpy ones.
    # each row is a data sample
    # each column is a feature

    # standardization
    standardized_data = np.copy(data)
    means = data.mean(axis=0)
    stds = data.std(axis=0)
    standardized_data = (standardized_data - means)/stds

    # minmax
    normalized_data = np.copy(data)
    mins = np.min(normalized_data, axis=0)
    maxes = np.max(normalized_data, axis=0)
    normalized_data = (normalized_data - mins)/(maxes-mins)

    return standardized_data, normalized_data
