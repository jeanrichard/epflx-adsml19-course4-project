# -*- coding: utf-8 -*-
"""\
Utilities for Course 4 - Project.
"""

# Standard library.
import os
import typing as T

# 3rd party.
import numpy as np


def load(path: os.PathLike) -> T.Mapping[str, np.ndarray]:
    """\
    DOCME
    """
    # Load the NPZ file.
    with np.load(path, allow_pickle=False) as npz_file:
        # Load all arrays.
        data = dict(npz_file.items())
    return data


def info(data: T.Mapping[str, np.ndarray]) -> str:
    """\
    DOCME
    """
    return '\n'.join([
        f'{key}: shape={val.shape}, dtype={val.dtype}'
        for key, val in data.items()
    ])
