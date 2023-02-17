#!/usr/bin/env python
from __future__ import annotations

from collections import UserDict, UserList
from typing import Iterable, overload, Any, Hashable, Iterator
from . import lib

try:
    from typing import Self
except ImportError:
    from typing import TypeVar

    Self = TypeVar("Self", bound="SliceDict")


class SliceDict(UserDict, dict):
    """
    A dict like container that support slicing, boolean selection
    and passing multiple keys at once.
    """

    # =====================================================
    # __get/set/del-item__
    # =====================================================

    def _expand_key(self: Self, key: Any) -> _SimpleIndex:
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
            key = _SimpleIndex(self.keys())[key]

        elif lib.is_list_like(key):
            # making a list from key, ensures numeric indexing
            # even if key has an index attribute that would be
            # used otherwise (e.g. with pd.Series)
            key = _SimpleIndex(key)

            if lib.is_boolean_indexer(key):
                if len(key) != len(self.keys()):
                    raise ValueError(
                        f"Boolean indexer has wrong length: "
                        f"{len(key)} instead of {len(self.keys())}"
                    )
                key = _SimpleIndex(k for i, k in enumerate(self.keys()) if key[i])
        else:
            raise TypeError(
                f"Key must be hashable, a list-like of hashable items, "
                f"a boolean list-like or a slice, not of type {type(key)}"
            )  # pragma: no cover
        return key

    @overload
    def __setitem__(self: Self, key: Hashable, value: Any) -> None:
        ...  # pragma: no cover

    @overload
    def __setitem__(self: Self, key: slice | Iterable, value: Iterable) -> None:
        ...  # pragma: no cover

    def __setitem__(self, key, value):
        """Sets a value or a collection of values."""

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
                f"Value must be some kind of "
                f"collection if multiple keys are given, "
                f"but is of type {type(value)}."
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

    def __setitem_single__(self: Self, key: Hashable, value: Any) -> None:
        return super().__setitem__(key, value)

    @overload
    def __getitem__(self: Self, key: Hashable) -> Any:
        ...  # pragma: no cover

    @overload
    def __getitem__(self: Self, key: slice | Iterable) -> Self:
        ...  # pragma: no cover

    def __getitem__(self, key):
        # scalar gives item, all other kinds of keys
        # return __class__ type instances (dict-likes)
        try:
            return super().__getitem__(key)
        except TypeError:  # not hashable
            pass

        key = self._expand_key(key)

        missing = key.difference(self.keys())
        if not missing.empty:
            raise KeyError(f"keys {missing.tolist()} does not exist")

        # INFO:
        # cannot call `super().method` in comprehensions
        return self.__class__(
            {k: super(self.__class__, self).__getitem__(k) for k in key}
        )

    # added to dict in py3.8
    def __reversed__(self: Self) -> Iterator:
        return reversed(self.keys())

    # added to dict in py3.9
    def __or__(self: Self, other: Self) -> Self:
        return self.__class__(self, **other)

    def __ror__(self: Self, other: Self) -> Self:
        return self.__class__(other, **self)

    # added to dict in py3.9
    def __ior__(self: Self, other: Self) -> Self:
        # return self, even if inplace
        data = dict(self.data)
        try:
            self.update(other)
            data = self.data
        finally:
            self.data = data
        return self


class TypedSliceDict(SliceDict):
    _key_types: tuple = ()
    _value_types: tuple = ()

    @property
    def value_types(self) -> tuple[type, ...]:
        return self._value_types or Any

    @property
    def key_types(self) -> tuple[type, ...]:
        return self._key_types or Hashable

    def __setitem_single__(self, key: Hashable, value: Any) -> None:
        if self._key_types:
            self._validate_type(key, self._key_types, "key", errors="raise")
        if self._value_types:
            self._validate_type(value, self._value_types, "value", errors="raise")
        super().__setitem_single__(key, value)

    @staticmethod
    def _validate_type(
        obj: object, types: type | tuple[type, ...], name: str, errors: str = "raise"
    ) -> bool:
        # errors: 'ignore' or 'raise'
        if isinstance(obj, types):
            return True
        if errors == "ignore":
            return False
        raise TypeError(
            f"{name} must be of type {' or '.join(map(repr, types))}, not {type(obj)}"
        )


class _SimpleIndex(UserList):
    def __init__(self, initlist: Iterable | None = None, dtype: type = None):
        super().__init__(initlist)
        self.dtype = dtype or getattr(initlist, "dtype", None)

    def __finalize__(self: _SimpleIndex, other: _SimpleIndex) -> _SimpleIndex:
        self.dtype = other.dtype
        return self

    def unique(self: _SimpleIndex) -> _SimpleIndex:
        # set() operations don't preserve order
        return _SimpleIndex(dict.fromkeys(self)).__finalize__(self)

    def difference(self: _SimpleIndex, other: Iterable) -> _SimpleIndex:
        # must be unique
        return _SimpleIndex(k for k in self.unique() if k not in other).__finalize__(
            self
        )

    def union(self: _SimpleIndex, other: Iterable) -> _SimpleIndex:
        # can have duplicates
        # to implement a full-fledged duplicate behavior, in the
        # sense that the index with most dupes win is quite tricky,
        # and seems impossible without value counting or successive
        # adding or removing items.
        # Index([1,1,2]).union(Index([1,2,2]) => Index([1,1,2,2])
        return (self + _SimpleIndex(k for k in other if k not in self)).__finalize__(
            self
        )

    def intersection(self: _SimpleIndex, other: Iterable) -> _SimpleIndex:
        # must be unique
        return _SimpleIndex(k for k in self.unique() if k in other).__finalize__(self)

    def symmetric_difference(self: _SimpleIndex, other: Iterable) -> _SimpleIndex:
        # must be unique
        return _SimpleIndex(
            k for k in self.union(other).unique() if k not in self.intersection(other)
        ).__finalize__(self)

    @property
    def empty(self: _SimpleIndex) -> bool:
        return len(self) == 0

    def tolist(self: _SimpleIndex) -> list:
        return list(self.data)  # shallow copy
