# -*- coding: utf-8 -*-
"""\
Utilities for Course 4 - Project.
"""

# Standard library.
import functools
import operator
import os
import pathlib
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


def _get_or_create_results(path: os.PathLike) -> pd.DataFrame:
    """\
    Returns a data-frame to hold the results of the various classifiers.
    """
    try:
        df_results = pd.read_csv(path, index_col='name')
    except FileNotFoundError as e:
        df_results = pd.DataFrame(data={
            'name': pd.Series(dtype=str),
            'desc': pd.Series(dtype=str),
            'test_acc': pd.Series(dtype=float)
        })
        df_results = df_results.set_index('name')
    return df_results


DEFAULT_RESULTS_FILENAME = pathlib.Path.cwd() / 'results.csv'


def persist_result(name: str, desc: str, test_acc: float, path: os.PathLike = DEFAULT_RESULTS_FILENAME) -> None:
    """\
    Updates the result for a given classifier (inserts a new row if needed).
    """
    df_results = _get_or_create_results(path)
    df_results.loc[name, ['desc', 'test_acc']] = (desc, test_acc)
    df_results.to_csv(path, header=True, index=True)


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
