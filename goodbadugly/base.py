#!/usr/bin/env python
from __future__ import annotations

import logging
from collections import UserDict
from functools import wraps, partialmethod

import pandas as pd
from pandas.core.common import is_bool_indexer
from typing import Mapping, MutableMapping, Iterator, Iterable, overload, Any, Hashable
from .lib import log_call


class BaseContainer(UserDict):
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)

    # =====================================================
    # Index
    # =====================================================

    def _check_key(self, key):  # noqa
        """Overwrite to restrict type of a single key"""
        return key

    def _check_value(self, value):  # noqa
        """Overwrite to restrict type of a single value"""
        return value

    @property
    def index(self):
        return pd.Index(self.keys())

    @index.setter
    def index(self, value):
        value = pd.Index(value)
        if len(self.keys()) != len(value):
            raise ValueError(
                f"Length mismatch: Expected {len(self.keys())} keys, "
                f"but got {len(value)} keys."
            )
        # we must expand the zip now, because values() are a view
        # and would change after clear()
        pairs = dict(zip(value, self.values()))
        self.clear()
        self.update(pairs)

    # =====================================================
    # __get/set/del-item__
    # =====================================================

    def _expand_key(self, key: Any) -> pd.Index:
        """
        Expand slice- and list-like and boolean list-likes to an index.

        Parameters
        ----------
        key : Any
            The key to expand.

        Returns
        -------
        index: pd.Index
            The expanded key

        Raises
        ------
        ValueError: If boolean list-like key missmatch
        TypeError: If key is a slice of other than integer
            or if key cannot be cast to pd.Index
        """
        if isinstance(key, slice):
            key = self.index[key]

        if is_bool_indexer(key):
            if len(key) != len(self.keys()):
                raise ValueError(
                    f"Unalienable boolean indexer. {len(self.keys())}"
                    f"items are present, but indexer is of length {len(key)}"
                )
            key = [k for i, k in enumerate(self.keys()) if key[i]]

        if not pd.api.types.is_list_like(key):
            raise TypeError(
                f"Cannot index with key of type {type(key).__name__}"
            )  # pragma: no cover

        if not isinstance(key, pd.Index):
            key = pd.Index(key)

        return key

    @overload
    def __setitem__(self, key: Hashable, value: Any) -> None:
        ...

    @overload
    def __setitem__(self, key: slice | Iterable, value: Iterable) -> None:
        ...

    def __setitem__(self, key, value):
        """
        Sets a value or a set of values at once.

        Parameters
        ----------
        key : hashable, slice of int, pd.Index, list-like or boolean list-like
            The key to index. If hashable (strings, numbers, None, etc.) the
            value of any type is
            value is
        value :

        Returns
        -------

        """
        if pd.api.types.is_hashable(key):
            key = self._check_key(key)
            value = self._check_value(value)
            super().__setitem__(key, value)
            return

        key = self._expand_key(key)

        # check value
        if pd.api.types.is_dict_like(value):
            value = [value[k] for k in value.keys()]
        if pd.api.types.is_iterator(value):
            value = list(value)
        if not pd.api.types.is_list_like(value):
            if len(key) == 1:
                value = [value]
            else:
                raise TypeError(
                    "value must be some kind of collection if multiple keys are given"
                )

        if len(key) != len(value):
            raise ValueError(
                f"Length mismatch: Got {len(key)} keys, "
                f"but value has {len(value)} items."
            )

        for i, val in enumerate(value):
            k = self._check_key(key[i])
            val = self._check_value(val)
            super().__setitem__(k, val)

    @overload
    def __getitem__(self, key: Hashable) -> Any:
        ...

    @overload
    def __getitem__(self, key: slice | Iterable) -> BaseContainer:
        ...

    def __getitem__(self, key):
        # scalar gives item, all other kinds of keys
        # return __class__ type instances (dict-likes)
        if pd.api.types.is_hashable(key):
            key = self._check_key(key)
            return super().__getitem__(key)

        key = self._expand_key(key)

        missing = key.difference(self.keys())
        if not missing.empty:
            raise KeyError(f"{missing.tolist()} does not exist")

        return self.__class__(
            {
                # cannot call super().method in comprehensions
                self._check_key(k): super(self.__class__, self).__getitem__(k)
                for k in key
            }
        )
