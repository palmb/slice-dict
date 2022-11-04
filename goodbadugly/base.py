#!/usr/bin/env python
from __future__ import annotations

from collections import UserDict

import pandas as pd
from pandas.core.common import is_bool_indexer
from typing import Iterable, overload, Any, Hashable


class _BaseContainer(UserDict):
    def _set_single_item_callback(self, key, value):  # noqa
        """
        Callback before setting a value for a single key.

        This is called *only* for single (hashable) keys.
        List-like, slices and other non-hashable keys are
        evaluated/expanded before and this callback is then
        called for each item within them.

        This callback can be used to restrict keys and/or
        values to specific types.

        Parameters
        ----------
        key : hashable
            The key to insert/replace a value.

        value : any
            The value to insert/replace.

        See Also
        --------

        Return
        ------
        key: hashable
        value: any
        """
        return key, value

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
            key = pd.Index(self.keys())[key]

        elif is_bool_indexer(key):
            if len(key) != len(self.keys()):
                raise ValueError(
                    f"Unalienable boolean indexer. {len(self.keys())}"
                    f"items are present, but indexer is of length {len(key)}"
                )
            key = [k for i, k in enumerate(self.keys()) if key[i]]

        elif pd.api.types.is_list_like(key):
            pass

        else:
            raise TypeError(
                f"Cannot index with key of type {type(key).__name__}"
            )  # pragma: no cover

        if not isinstance(key, pd.Index):
            key = pd.Index(key)

        return key

    @overload
    def __setitem__(self, key: Hashable, value: Any) -> None:
        ...  # pragma: no cover

    @overload
    def __setitem__(self, key: slice | Iterable, value: Iterable) -> None:
        ...  # pragma: no cover

    def __setitem__(self, key, value):
        """Sets a value or a set of values at once."""

        _cb = self._set_single_item_callback

        if pd.api.types.is_hashable(key):
            return super().__setitem__(*_cb(key, value))
            # return self.__setitem_single_key__(key, value)

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

        for k, val in zip(key, value):
            if not pd.api.types.is_hashable(k):
                raise KeyError(f"key '{k}' is not hashable")
            super().__setitem__(*_cb(k, val))

    @overload
    def __getitem__(self, key: Hashable) -> Any:
        ...  # pragma: no cover

    @overload
    def __getitem__(self, key: slice | Iterable) -> _BaseContainer:
        ...  # pragma: no cover

    def __getitem__(self, key):
        # scalar gives item, all other kinds of keys
        # return __class__ type instances (dict-likes)
        if pd.api.types.is_hashable(key):
            return super().__getitem__(key)

        key = self._expand_key(key)

        missing = key.difference(self.keys())
        if not missing.empty:
            raise KeyError(f"{missing.tolist()} does not exist")

        # INFO:
        # cannot call `super().method` in comprehensions
        return self.__class__(
            {k: super(self.__class__, self).__getitem__(k) for k in key}
        )


class _Axis:
    def __init__(self, name):
        self._name = name

    def __get__(self, instance, owner):
        if instance is None:  # class attribute access
            return self
        return pd.Index(instance.keys())

    def __set__(self, instance, value):
        value = pd.Index(value)
        if len(instance.keys()) != len(value):
            raise ValueError(
                f"Length mismatch: Expected {len(instance.keys())} "
                f"{self._name} keys, but got {len(value)} keys."
            )
        # we must expand the zip now, because values() are a view
        # and would change after clear()
        pairs = dict(zip(value, instance.values()))
        instance.clear()
        instance.update(pairs)


class IndexContainer(_BaseContainer):
    index = _Axis("index")


class ColumnContainer(_BaseContainer):
    columns = _Axis("columns")
