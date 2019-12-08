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
from sklearn.utils import class_weight


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


def reset_seeds(seed: int = 0) -> None:
    """\
    Sets the seed of multiple PRNG's to a given value.
    
    .. seealso:: https://medium.com/@ODSC/properly-setting-the-random-seed-in-ml-experiments-not-as-simple-as-you-might-imagine-219969c84752
    
    Args:
        seed: A given value. 
    """
    # 1. Set ``PYTHONHASHSEED`` environment variable to a fixed value.
    import os
    os.environ['PYTHONHASHSEED']=str(seed)
    
    # 2. Set Python's built-in PRNG seed seed to a fixed value.
    import random
    random.seed(seed)
    
    # 3. Set NumPy's PRNG seed to a fixed value.
    import numpy as np
    np.random.seed(seed)
    
    # 4. Set TensorFlow's PRNG seed to a fixed value.
    import tensorflow as tf
    tf.set_random_seed(seed)


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


def get_class_weight(y: np.ndarray) -> T.Dict[T.Any, float]:
    """\
    DOCME
    """
    y_classes = np.unique(y)
    class_weight_ = dict(zip(y_classes, class_weight.compute_class_weight('balanced', y_classes, y)))
    return class_weight_
