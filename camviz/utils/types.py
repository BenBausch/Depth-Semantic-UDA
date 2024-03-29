# Copyright 2020 Toyota Research Institute.  All rights reserved.

import numpy as np
import torch


def is_numpy(data):
    """Checks if data is a numpy array."""
    return isinstance(data, np.ndarray)


def is_tensor(data):
    """Checks if data is a torch tensor."""
    return type(data) == torch.Tensor


def is_tuple(data):
    """Checks if data is a tuple."""
    return isinstance(data, tuple)


def is_list(data):
    """Checks if data is a list."""
    return isinstance(data, list)


def is_double_list(data):
    """Checks if data is a double list (list of lists)"""
    return is_list(data) and is_list(data[0])


def is_dict(data):
    """Checks if data is a dictionary."""
    return isinstance(data, dict)


def is_str(data):
    """Checks if data is a string."""
    return isinstance(data, str)


def is_int(data):
    """Checks if data is an integer."""
    return isinstance(data, int)


def is_float(data):
    """Checks if data is a float value."""
    return isinstance(data, float)


def is_seq(data):
    """Checks if data is a list or tuple."""
    return is_tuple(data) or is_list(data)

