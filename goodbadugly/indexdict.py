#!/usr/bin/env python

import pandas as pd
from pandas.core.common import is_bool_indexer


class BaseContainer(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__update_index()

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
        pairs = dict(zip(value, self.values()))
        self.clear()
        self.update(pairs)
        self.__update_index()

    def _slice_to_index(self, sl) -> pd.Index:
        return self._index[sl]

    def _expand_key(self, key) -> pd.Index:
        if isinstance(key, slice):
            key = self._slice_to_index(key)
        if is_bool_indexer(key):
            if len(key) != len(self.keys()):
                raise ValueError(
                    f"Unalienable boolean indexer. {len(self.keys())}"
                    f"items are present, but indexer is of length {len(key)}"
                )
            key = [k for i, k in enumerate(self.keys()) if key[i]]
        if not pd.api.types.is_list_like(key):
            raise TypeError(f"Cannot index with key of type {type(key).__name__}")
        if not isinstance(key, pd.Index):
            key = pd.Index(key)
        return key

    def __setitem__(self, key, value):
        if pd.api.types.is_hashable(key):
            super().__setitem__(key, value)
            self.__update_index()
            return

        key = self._expand_key(key)

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
        for i, k in enumerate(key):
            super().__setitem__(k, value[i])
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

        # cannot call super() in comprehensions
        return self.__class__({k: dict.__getitem__(self, k) for k in key})

    def __delitem__(self, key):
        super().__delitem__(key)
        self.__update_index()
