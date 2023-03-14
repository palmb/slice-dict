#!/usr/bin/env python
from __future__ import annotations

import pytest
from sliceable_dict.core import _SimpleIndex


class MinimalListLike:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]


class DtypedLL(MinimalListLike):
    dtype = bool


@pytest.mark.parametrize("value", [[], (), {}, [1, 2], MinimalListLike([1, 2])])
def test_construct(value):
    result = _SimpleIndex(value)
    assert isinstance(result, _SimpleIndex)
    assert list(result) == list(value)
    assert result == list(value)


@pytest.mark.parametrize(
    "value, expected",
    [
        ([1], None),
        (MinimalListLike([]), None),
        (MinimalListLike([]), None),
        (DtypedLL([]), bool),
    ],
)
def test_dtype(value, expected):
    result = _SimpleIndex(value)
    assert isinstance(result, _SimpleIndex)
    assert result.dtype == expected


@pytest.mark.parametrize("name", dir(list))
def test_list_attrs(name):
    assert hasattr(_SimpleIndex, name)


@pytest.mark.parametrize(
    "name,args,expected",
    [
        ("append", (99,), [0, 99]),
        ("insert", (0, 99), [99, 0]),
        ("insert", (1, 99), [0, 99]),
        ("insert", (10, 99), [0, 99]),
        ("insert", (-10, 99), [99, 0]),
        ("extend", (_SimpleIndex([]),), [0]),
        ("extend", (_SimpleIndex([1]),), [0, 1]),
        ("extend", (_SimpleIndex([1, 2]),), [0, 1, 2]),
        ("clear", (), []),
    ],
)
def test_setter(name, args, expected):
    inst = _SimpleIndex([0])
    result = getattr(inst, name)(*args)
    assert isinstance(result, _SimpleIndex)
    assert result == expected


@pytest.mark.parametrize(
    "name,args,expected",
    [
        ("append", (99,), "result"),
        ("insert", (0, 99), "result"),
        ("extend", (_SimpleIndex([]),), "result"),
        ("extend", (_SimpleIndex([1, 2]),), "result"),
        ("clear", (), "result"),
        ("__delitem__", (0,), "error"),
        ("__setitem__", (0, 99), "error"),
    ],
)
def test_immutable(name, args, expected):
    inst = _SimpleIndex([0])
    func = getattr(inst, name)
    if expected == "result":
        result = func(*args)
        assert isinstance(result, _SimpleIndex)
        assert result is not inst
    else:
        with pytest.raises(TypeError):
            func(*args)


@pytest.mark.parametrize(
    "value, expected",
    [
        ([], True),
        (["a"], True),
        (["a", "b"], True),
        (["a", "a"], False),
    ],
)
def test_is_unique(value, expected):
    result = _SimpleIndex(value).is_unique
    assert result == expected


def test_empty():
    assert _SimpleIndex().empty is True
    assert _SimpleIndex(["a"]).empty is False
    assert _SimpleIndex([None]).empty is False
    assert _SimpleIndex(["a", "b"]).empty is False


@pytest.mark.parametrize(
    "value", [[], ["a", "b"], ["a", "a"], ["a", "a", "b"], ["a", "b", "a"]]
)
def test_tolist(value):
    result = _SimpleIndex(value).tolist()
    assert isinstance(result, list)
    assert result == value


@pytest.mark.parametrize(
    "value, expected",
    [
        ([], []),
        (["a", "b"], ["a", "b"]),
        (["a", "a"], ["a"]),
        (["a", "a", "b"], ["a", "b"]),
        (["a", "b", "a"], ["a", "b"]),
    ],
)
def test_unique(value, expected):
    result = _SimpleIndex(value).unique()
    assert result == expected


@pytest.mark.parametrize(
    "left,right,expected",
    [
        ([], [], []),
        (["a"], [], ["a"]),
        ([], ["a"], ["a"]),
        (["a"], ["a"], ["a"]),
        (["a"], ["b"], ["a", "b"]),
        (["b"], ["a"], ["b", "a"]),
        (["a", "b"], ["b"], ["a", "b"]),
        (["a", "b"], ["c"], ["a", "b", "c"]),
        (["c"], ["a", "b"], ["c", "a", "b"]),
    ],
)
def test_union(left, right, expected):
    # left | right
    result = _SimpleIndex(left).union(right)
    assert isinstance(result, _SimpleIndex)
    assert result == expected


@pytest.mark.parametrize(
    "left,right,expected",
    [
        ([], [], []),
        (["a"], [], []),
        ([], ["a"], []),
        (["a"], ["a"], ["a"]),
        (["a"], ["b"], []),
        (["a", "b"], ["b"], ["b"]),
        (["a", "b"], ["c"], []),
        (["a", "b"], ["a", "c"], ["a"]),
        (["a", "b"], ["c", "a"], ["a"]),
        (["a", "b"], ["b", "a"], ["a", "b"]),
        (["b", "a"], ["a", "b"], ["b", "a"]),
        (["a", "a"], ["a", "a", "a"], ["a"]),
    ],
)
def test_intersection(left, right, expected):
    # left & right
    result = _SimpleIndex(left).intersection(right)
    assert isinstance(result, _SimpleIndex)
    assert result == expected


@pytest.mark.parametrize(
    "left,right,expected",
    [
        ([], [], []),
        (["a"], [], ["a"]),
        ([], ["a"], []),
        (["a"], ["a"], []),
        (["a"], ["b"], ["a"]),
        (["a"], ["a", "b"], []),
        (["a"], ["b", "c"], ["a"]),
        (["a", "b"], ["b"], ["a"]),
        (["a", "b"], ["c"], ["a", "b"]),
        (["a", "b"], ["a", "c"], ["b"]),
        (["a", "b"], ["c", "a"], ["b"]),
        (["a", "b"], ["b", "a"], []),
        (["b", "a"], ["a", "b"], []),
        (["a", "a"], ["a", "a", "a"], []),
        (["a", "a"], [], ["a"]),
    ],
)
def test_difference(left, right, expected):
    # left - right
    result = _SimpleIndex(left).difference(right)
    assert isinstance(result, _SimpleIndex)
    assert result == expected


@pytest.mark.parametrize(
    "left,right,expected",
    [
        ([], [], []),
        (["a"], [], ["a"]),
        ([], ["a"], ["a"]),
        (["a"], ["a"], []),
        (["a"], ["b"], ["a", "b"]),
        (["a"], ["a", "b"], ["b"]),
        (["a"], ["b", "c"], ["a", "b", "c"]),
        (["a", "b"], ["b"], ["a"]),
        (["a", "b"], ["c"], ["a", "b", "c"]),
        (["a", "b"], ["a", "c"], ["b", "c"]),
        (["a", "b"], ["c", "a"], ["b", "c"]),
        (["a", "b"], ["b", "a"], []),
        (["b", "a"], ["a", "b"], []),
        (["a", "a"], ["a", "a", "a"], []),
        (["a", "a"], [], ["a"]),
    ],
)
def test_symmetric_difference(left, right, expected):
    # left ^ right
    result = _SimpleIndex(left).symmetric_difference(right)
    assert isinstance(result, _SimpleIndex)
    assert result == expected
