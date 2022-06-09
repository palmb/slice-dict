#!/usr/bin/env python
from __future__ import annotations

from collections import UserDict
from functools import wraps, partialmethod

import pandas as pd
from pandas.core.common import is_bool_indexer
from typing import Mapping, MutableMapping, Iterator, Iterable


class BaseContainer(UserDict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__update_index()

    # =====================================================
    # Index
    # =====================================================

    def __update_index(self):
        self._index = pd.Index(self.keys())

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, value):
        value = pd.Index(value)
        if len(self._index) != len(value):
            raise ValueError(
                f"Length mismatch: Expected {len(self._index)} keys, "
                f"but got {len(value)} keys."
            )
        # we must expand the zip now, because values() are a view
        # and would change after clear()
        pairs = dict(zip(value, self.values()))
        self.clear()
        self.update(pairs)
        # self.__update_index()

    # =====================================================
    # __get/set/del-item__
    # =====================================================

    def _expand_key(self, key) -> pd.Index:
        if isinstance(key, slice):
            key = self._index[key]
        if is_bool_indexer(key):
            if len(key) != len(self.keys()):
                raise ValueError(
                    f"Unalienable boolean indexer. {len(self.keys())}"
                    f"items are present, but indexer is of length {len(key)}"
                )
            key = [k for i, k in enumerate(self.keys()) if key[i]]
        if not pd.api.types.is_list_like(key):
            raise TypeError(f"Cannot index with key of type {type(key).__name__}")  # pragma: no cover
        if not isinstance(key, pd.Index):
            key = pd.Index(key)
        return key

    def __setitem__(self, key, value):
        if pd.api.types.is_hashable(key):
            super().__setitem__(key, value)
            self.__update_index()
            return

        key = self._expand_key(key)

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
            super().__setitem__(key[i], val)
        self.__update_index()

    def __getitem__(self, key):
        # scalar gives item, all other kinds of keys
        # return __class__ type instances (dict-likes)
        if pd.api.types.is_hashable(key):
            return super().__getitem__(key)

        key = self._expand_key(key)
        missing = key.difference(self.keys())
        if not missing.empty:
            raise KeyError(f"{missing.tolist()} does not exist")

        # cannot call super().method in comprehensions
        return self.__class__(
            {k: super(self.__class__, self).__getitem__(k) for k in key}
        )

    def __delitem__(self, key):
        super().__delitem__(key)
        self.__update_index()

    # =====================================================
    # dict methods
    # =====================================================

    # no need to overwrite
    # --------------------
    # def clear(self) -> None:
    # def copy(self) -> BaseContainer:
    # def fromkeys(cls, __iterable: Iterable, __value=None) -> BaseContainer:
    # def popitem(self) -> tuple:
    # def update(self, __m: Mapping | None = None, **kwargs) -> None:

    def __ior__(self, other):
        # inplace-or, returns a modified self AND modifies
        _self = super().__ior__(other)
        assert _self is self
        self.__update_index()
        return self



