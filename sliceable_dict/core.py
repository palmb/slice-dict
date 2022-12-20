#!/usr/bin/env python
from __future__ import annotations

from collections import UserDict, UserList
from typing import Iterable, overload, Any, Hashable, final, Tuple

from . import lib


# todo: spead the `final()` word to the world


class SimpleIndex(UserList):
    def unique(self) -> SimpleIndex:
        # set() operations don't preserve order
        return SimpleIndex(dict.fromkeys(self))

    def difference(self, other: Iterable) -> SimpleIndex:
        # must be unique
        return SimpleIndex(k for k in self.unique() if k not in other)

    def union(self, other: Iterable) -> SimpleIndex:
        # can have duplicates
        # to implement a full-fledged duplicate behavior, in the
        # sense that the index with most dupes win is quite tricky,
        # and seems impossible without value counting or successive
        # adding or removing items.
        # Index([1,1,2]).union(Index([1,2,2]) => Index([1,1,2,2])
        return self + SimpleIndex(k for k in other if k not in self)

    def intersection(self, other: Iterable) -> SimpleIndex:
        # must be unique
        return SimpleIndex(k for k in self.unique() if k in other)

    def symmetric_difference(self, other: Iterable) -> SimpleIndex:
        # must be unique
        return SimpleIndex(
            k for k in self.union(other).unique() if k not in self.intersection(other)
        )

    @property
    def empty(self) -> bool:
        return len(self) == 0

    def tolist(self) -> list:
        return self.data


class SliceDict(UserDict, dict):
    """
    A dict like container that support slicing, boolean selection
    and passing multiple keys at once.
    """

    # =====================================================
    # __get/set/del-item__
    # =====================================================

    def _expand_key(self, key: Any) -> SimpleIndex:
        """
        Expand slice- and list-like and boolean list-likes to an index.

        Parameters
        ----------
        key : Any
            The key to expand.

        Returns
        -------
        index: Index
            The expanded key

        Raises
        ------
        ValueError: If boolean list-like key missmatch
        TypeError: If key is a slice of other than integer
            or if key cannot be cast to Index
        """
        if isinstance(key, slice):
            key = SimpleIndex(self.keys())[key]
        elif lib.is_list_like(key):
            if lib.is_boolean_indexer(key):
                if len(key) != len(self.keys()):
                    raise ValueError(
                        f"Boolean indexer has wrong length: "
                        f"{len(key)} instead of {len(self.keys())}"
                    )
                key = SimpleIndex(k for i, k in enumerate(self.keys()) if key[i])
            elif not isinstance(key, SimpleIndex):
                key = SimpleIndex(key)
        else:
            raise TypeError(
                f"Key must be hashable, a slice or a list-like of hashable items, "
                f"but type {type(key).__name__} was given."
            )  # pragma: no cover
        return key

    @overload
    def __setitem__(self, key: Hashable, value: Any) -> None:
        ...  # pragma: no cover

    @overload
    def __setitem__(self, key: slice | Iterable, value: Iterable) -> None:
        ...  # pragma: no cover

    def __setitem__(self, key, value):
        """Sets a value or a collection of values."""

        # includes generators, range(3), etc.
        if lib.is_hashable(key):
            return self.__setitem_single__(key, value)
        key = self._expand_key(key)

        if lib.is_dict_like(value):
            value = [value[k] for k in value.keys()]
        elif lib.is_list_like(value):
            pass
        elif len(key) == 1:
            value = [value]
        else:
            raise TypeError(
                "Value must be some kind of collection if multiple keys are given,"
                f"but {type(value).__name__} is not."
            )
        if len(key) != len(value):
            raise ValueError(f"Got {len(key)} keys, but {len(value)} values.")

        data = dict(self.data)  # shallow copy
        try:
            for k, v in zip(key, value):
                self.__setitem_single__(k, v)
            data = self.data
        finally:
            self.data = data

    def __setitem_single__(self, key: Hashable, value: Any):
        return super().__setitem__(key, value)

    @overload
    def __getitem__(self, key: Hashable) -> Any:
        ...  # pragma: no cover

    @overload
    def __getitem__(self, key: slice | Iterable) -> SliceDict:
        ...  # pragma: no cover

    def __getitem__(self, key):
        # scalar gives item, all other kinds of keys
        # return __class__ type instances (dict-likes)
        if lib.is_hashable(key):
            return super().__getitem__(key)

        key = self._expand_key(key)

        missing = key.difference(self.keys())
        if not missing.empty:
            raise KeyError(f"keys {missing.tolist()} does not exist")

        # INFO:
        # cannot call `super().method` in comprehensions
        return self.__class__(
            {k: super(self.__class__, self).__getitem__(k) for k in key}
        )


class TypedSliceDict(SliceDict):

    _key_types: tuple = ()
    _value_types: tuple = ()

    @final
    def __setitem_single__(self, key: Hashable, value: Any):
        if self._key_types and not isinstance(key, self._key_types):
            key = self._convert_key(key)
            if not isinstance(key, self._key_types):
                raise TypeError(
                    f"Key must be of type {'or'.join(self._key_types)}, "
                    f"not {type(key).__name__}"
                )
        if self._value_types and not isinstance(value, self._value_types):
            value = self._convert_value(value)
            if not isinstance(value, self._value_types):
                raise TypeError(
                    f"Value must be of type {'or'.join(self._value_types)}, "
                    f"not {type(value).__name__}"
                )
        super().__setitem_single__(key, value)

    def _convert_key(self, key: Hashable) -> Hashable:  # noqa
        return key

    def _convert_value(self, value: Any) -> Any:  # noqa
        return value
