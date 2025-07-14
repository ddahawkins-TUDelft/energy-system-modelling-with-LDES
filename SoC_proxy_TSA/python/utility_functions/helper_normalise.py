import pandas as pd
import numpy as np

def normalise_by_method(data, method="mean", inplace=False):
    """
    Normalizes the input (list, dict, or pandas Series) using one of:
    - method="sum": values sum to 1
    - method="mean": values are scaled by mean
    - method="max": values are scaled by max
    
    Returns an object of the same type. Supports inplace modification for dicts, lists, Series.
    """
    if method not in {"sum", "mean", "max"}:
        raise ValueError("Method must be 'sum', 'mean', or 'max'")

    # Convert to pandas Series for unified processing
    if isinstance(data, dict):
        s = pd.Series(data)
    elif isinstance(data, (list, np.ndarray)):
        s = pd.Series(data)
    elif isinstance(data, pd.Series):
        s = data
    else:
        raise TypeError("Input must be a list, dictionary, or pandas Series.")

    # Choose normalization factor
    if method == "sum":
        factor = s.sum()
    elif method == "mean":
        factor = s.mean()
    elif method == "max":
        factor = s.max()

    # Avoid divide-by-zero errors
    if factor == 0:
        raise ValueError("Normalization factor is zero — cannot normalize.")

    normed = s / factor

    # Return in same format
    if isinstance(data, dict):
        if inplace:
            data.update(normed.to_dict())
            return data
        else:
            return normed.to_dict()

    elif isinstance(data, list) or isinstance(data, np.ndarray):
        normed_list = normed.tolist()
        if inplace:
            data[:] = normed_list
            return data
        else:
            return normed_list

    elif isinstance(data, pd.Series):
        if inplace:
            data[:] = normed
            return data
        else:
            return normed