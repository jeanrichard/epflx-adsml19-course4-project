# -*- coding: utf-8 -*-
"""\
Utilities for Course 4 - Project.
"""

# Standard library.
import functools
import operator
import os
import typing as T

# 3rd party.
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


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


def filter_like(df: pd.DataFrame, idx: T.Any, columns: T.Sequence[str]) -> pd.DataFrame:
    """\
    DOCME
    """
    # We need to be careful: None values are never equal to other values in Pandas.
    def eq(series: pd.Series, value: T.Optional[T.Any]) -> pd.Series:
        return series.isna() if value is None else series == value

    conditions = [
        eq(df[column], df.loc[idx, column]) for column in columns
    ]

    is_like = functools.reduce(operator.and_, conditions)

    return df[is_like]
