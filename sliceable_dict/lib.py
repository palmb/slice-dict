#!/usr/bin/env python

from typing import Any, Hashable
from collections import abc


# taken from
# https://github.com/pandas-dev/pandas/blob/v1.5.2/pandas/core/dtypes/inference.py
def is_hashable(obj: Any) -> bool:
    """
    Return True if hash(obj) will succeed, False otherwise.
    Some types will pass a test against collections.abc.Hashable but fail when
    they are actually hashed with hash().
    Distinguish between these and other types by trying the call to hash() and
    seeing if they raise TypeError.
    Returns
    -------
    bool
    Examples
    --------
    >>> import collections
    >>> a = ([],)
    >>> isinstance(a, collections.abc.Hashable)
    True
    >>> is_hashable(a)
    False
    """
    # Unfortunately, we can't use isinstance(obj, collections.abc.Hashable),
    # which can be faster than calling hash. That is because numpy scalars
    # fail this test.

    # Reconsider this decision once this numpy bug is fixed:
    # https://github.com/numpy/numpy/issues/5562
    try:
        hash(obj)
    except TypeError:
        return False
    else:
        return True


# taken from:
# https://github.com/pandas-dev/pandas/blob/v1.5.2/pandas/core/dtypes/inference.py
def is_dict_like(obj: Any) -> bool:
    """
        Check if the object is dict-like.
        Parameter    def is_hashable(obj):
            try:
                hash(obj)
            except TypeError:
                return False
            else:
                return True
    s
        ----------
        obj : The object to check
        Returns
        -------
        is_dict_like : bool
            Whether `obj` has dict-like properties.
        Examples
        --------
        >>> is_dict_like({1: 2})
        True
        >>> is_dict_like([1, 2, 3])
        False
        >>> is_dict_like(dict)
        False
        >>> is_dict_like(dict())
        True
    """
    if isinstance(obj, dict):
        return True
    dict_like_attrs = ("__getitem__", "keys", "__contains__")
    return (
        all(hasattr(obj, attr) for attr in dict_like_attrs)
        # [GH 25196] exclude classes
        and not isinstance(obj, type)
    )


# taken and modified from
# https://github.com/pandas-dev/pandas/blob/v1.5.2/pandas/_libs/lib.pyx
def is_list_like(obj: Any, allow_sets: bool = True):
    """
    Check if the object is list-like.

        Objects that are considered list-like are for example Python
        lists, tuples, sets, NumPy arrays, and Pandas Series.

        Strings and datetime objects, however, are not considered list-like.

        Parameters
        ----------
        obj : object
            Object to check.
        allow_sets : bool, default True
            If this parameter is False, sets will not be considered list-like.

        Returns
        -------
        bool
            Whether `obj` has list-like properties.

        Examples
        --------
        >>> import datetime
        >>> import numpy as np
        >>> is_list_like([1, 2, 3])
        True
        >>> is_list_like({1, 2, 3})
        True
        >>> is_list_like(datetime.datetime(2017, 1, 1))
        False
        >>> is_list_like("foo")
        False
        >>> is_list_like(1)
        False
        >>> is_list_like(np.array([2]))
        True
        >>> is_list_like(np.array(2))
        False
    """
    # first, performance short-cuts for the most common cases
    if isinstance(obj, list):
        return True
    # then the generic implementation
    return (
        # equiv:  getattr(obj, "__iter__", None) is not None and not isinstance(obj, type)
        isinstance(obj, abc.Iterable)
        # we do not count strings/unicode/bytes as list-like
        and not isinstance(obj, (str, bytes))
        # exclude zero-dimensional duck-arrays, effectively scalars
        and not (hasattr(obj, "ndim") and obj.ndim == 0)
        # exclude sets if allow_sets is False
        and not (allow_sets is False and isinstance(obj, abc.Set))
    )


def is_boolean_indexer(obj: abc.Iterable) -> bool:
    """
    Check whether `obj` is a valid boolean indexer.

    This assumes that obj is iterable !

    Parameters
    ----------
    obj : Any
        Only list-likes may be considered as valid
        boolean indexer. At least `obj` must be iterable.

    Returns
    -------
    bool
        Whether `obj` is valid boolean indexer.
    """
    return (
        hasattr(obj, "dtype")
        and obj.dtype == bool
        # (return first non-boolean element or return True) is True
        or next(filter(lambda e: not isinstance(e, bool), obj), True) is True
    )
