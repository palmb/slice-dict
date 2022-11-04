#!/usr/bin/env python
import numpy as np
import pytest
import pandas as pd
from operator import or_
from goodbadugly.base import IndexContainer, ColumnContainer, _BaseContainer

T, F = True, False


@pytest.mark.parametrize("klass", [ColumnContainer, IndexContainer])
@pytest.mark.parametrize("kwargs", [dict(), dict(a=99), dict(x=99)])
@pytest.mark.parametrize("args", [(), ([[1, 2]],), (dict(a="a", b="b"),)])
def test_creation(klass, args, kwargs):
    bc = klass(*args, **kwargs)
    assert isinstance(bc, klass)
    assert isinstance(bc, _BaseContainer)
    # assert isinstance(bc, dict)


@pytest.mark.parametrize("klass", [ColumnContainer, IndexContainer])
@pytest.mark.parametrize("attr", dir(dict))
def test_attrs(klass, attr):
    assert hasattr(klass, attr)
    if issubclass(klass, ColumnContainer):
        assert hasattr(klass, "columns")
    if issubclass(klass, IndexContainer):
        assert hasattr(klass, "index")


@pytest.mark.parametrize(
    "klass,axis_name", [(ColumnContainer, "columns"), (IndexContainer, "index")]
)
@pytest.mark.parametrize("key", [None, 1, 1.0, "a", b"a", np.nan])
def test_axis(klass, axis_name, key):
    bc = klass(zzz=None)
    assert getattr(bc, axis_name).equals(pd.Index(["zzz"]))
    bc[key] = None
    assert getattr(bc, axis_name).equals(pd.Index(["zzz", key]))
    del bc[key]
    assert getattr(bc, axis_name).equals(pd.Index(["zzz"]))


@pytest.mark.parametrize(
    "klass,axis_name", [(ColumnContainer, "columns"), (IndexContainer, "index")]
)
@pytest.mark.parametrize(
    "setter_name,args,expected",
    [
        # `{"a": None}` is in the container per default
        ("__copy__", (), pd.Index(["a"])),
        ("__or__", ({"b": 1},), pd.Index(["a", "b"])),
        ("__ror__", ({"b": 1},), pd.Index(["b", "a"])),
        ("__ior__", ({"b": 1},), pd.Index(["a", "b"])),  # works inplace and return self
        ("fromkeys", (["b"],), pd.Index(["b"])),
        ("copy", (), pd.Index(["a"])),
    ],
)
def test_index_update__methods_with_result(
    klass, axis_name, setter_name, args, expected
):
    bc = klass(a=None)
    result = getattr(bc, setter_name)(*args)
    assert isinstance(result, _BaseContainer)
    assert isinstance(result, klass)
    assert getattr(result, axis_name).equals(expected)


@pytest.mark.parametrize(
    "klass,axis_name", [(ColumnContainer, "columns"), (IndexContainer, "index")]
)
@pytest.mark.parametrize(
    "setter_name,args,expected",
    [
        # `{"a": None}` is in the container per default
        ("__setitem__", ("b", 1), pd.Index(["a", "b"])),
        ("__delitem__", ("a",), pd.Index([])),
        ("__ior__", ({"b": 1},), pd.Index(["a", "b"])),  # works inplace and return self
        ("setdefault", ("b", 1), pd.Index(["a", "b"])),
        ("setdefault", ("a", 1), pd.Index(["a"])),
        ("pop", ("a",), pd.Index([])),
        ("pop", ("b", None), pd.Index(["a"])),
        ("popitem", (), pd.Index([])),
        ("update", ({"b": 1},), pd.Index(["a", "b"])),
        ("clear", (), pd.Index([])),
    ],
)
def test_index_update__inplace_methods(klass, axis_name, setter_name, args, expected):
    bc = klass(a=None)
    getattr(bc, setter_name)(*args)
    assert getattr(bc, axis_name).equals(expected)


@pytest.mark.parametrize(
    "klass,axis_name", [(ColumnContainer, "columns"), (IndexContainer, "index")]
)
def test_index_setter(klass, axis_name):
    bc = klass(a=10, b=20, c=30)
    setattr(bc, axis_name, [1, 2, 3])
    assert bc.keys() == dict.fromkeys([1, 2, 3]).keys()
    with pytest.raises(ValueError):
        setattr(bc, axis_name, [1, 2])
