#!/usr/bin/env python
import numpy as np
import pytest
import pandas as pd
from operator import or_
from goodbadugly import BaseContainer

T, F = True, False


class AnyChild(BaseContainer):
    pass


@pytest.fixture(params=[BaseContainer, AnyChild])
def container_or_child(request):
    return request.param


@pytest.mark.parametrize("kwargs", [dict(), dict(a=99), dict(x=99)])
@pytest.mark.parametrize("args", [(), ([[1, 2]],), (dict(a="a", b="b"),)])
def test_creation(container_or_child, args, kwargs):
    bc = container_or_child(*args, **kwargs)
    assert isinstance(bc, container_or_child)
    assert isinstance(bc, BaseContainer)
    # assert isinstance(bc, dict)


@pytest.mark.parametrize("attr", dir(dict))
def test_attrs(container_or_child, attr):
    assert hasattr(container_or_child, attr)


_test_values = [
    None,
    1,
    "string",
    ["0", 1, 2.0],
    {"key": "value"},
    BaseContainer(x="x"),
    AnyChild(x="x"),
    pd.Series(index=[1, 2], dtype=float),
    pd.DataFrame(1, index=[1, 2], columns=["c0", "c1"]),
]


@pytest.mark.parametrize("key", [None, 1, 1.0, "a", b"a", np.nan])
def test_index(container_or_child, key):
    bc = container_or_child(zzz=None)
    assert bc.index.equals(pd.Index(["zzz"]))
    bc[key] = None
    assert bc.index.equals(pd.Index(["zzz", key]))
    del bc[key]
    assert bc.index.equals(pd.Index(["zzz"]))


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
def test_update__methods_with_result(container_or_child, setter_name, args, expected):
    bc = container_or_child(a=None)
    result = getattr(bc, setter_name)(*args)
    assert isinstance(result, container_or_child), type(result)
    assert result.index.equals(expected)


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
def test_update_index_inplace_methods(container_or_child, setter_name, args, expected):
    bc = container_or_child(a=None)
    getattr(bc, setter_name)(*args)
    assert bc.index.equals(expected)


def test_index_setter(container_or_child):
    bc = container_or_child(a=10, b=20, c=30)
    bc.index = [1, 2, 3]
    assert bc.keys() == dict.fromkeys([1, 2, 3]).keys()
    with pytest.raises(ValueError):
        bc.index = ["a", "b"]


@pytest.mark.parametrize("value", _test_values)
def test_values(container_or_child, value):
    bc = container_or_child(a=value)
    assert isinstance(bc["a"], type(value))


@pytest.mark.parametrize("key", [None, 1, 1.0, "a", b"a", np.nan])
def test_keys(container_or_child, key):
    bc = container_or_child()
    bc[key] = None
    assert key in bc.keys()


@pytest.mark.parametrize(
    "key,expected",
    [
        # lists
        (["a"], dict(a=0)),
        (["a", "b"], dict(a=0, b=0)),
        (pd.Index(["a"]), dict(a=0)),
        (pd.Index(["a", "b"]), dict(a=0, b=0)),
        # slices
        (slice(None), dict(a=0, b=0, c=0)),
        (slice(None, 10), dict(a=0, b=0, c=0)),
        (slice(0, 2), dict(a=0, b=0)),  # exclusive right bound
        (slice(1, 1), dict()),  # exclusive right bound
        (slice(None, None, 2), dict(a=0, c=0)),
        # bool
        ([F, F, F], dict()),
        ([T, F, F], dict(a=0)),
        ([T, F, T], dict(a=0, c=0)),
        ([T, T, T], dict(a=0, b=0, c=0)),
    ],
)
def test__getitem__complex_keys(container_or_child, key, expected):
    bc = container_or_child(a=0, b=0, c=0)
    result = bc[key]
    assert isinstance(result, BaseContainer)

    expected = container_or_child(expected)
    assert result.keys() == expected.keys()
    assert result.index.equals(expected.index)


@pytest.mark.parametrize(
    "key,err,msg",
    [
        ("x", KeyError, "x"),
        (["a", "y"], KeyError, "['y'] does not exist"),
        (["x", "y"], KeyError, "['x', 'y'] does not exist"),
        ([T, F], ValueError, r"Unalienable boolean indexer."),
        ([T, F, F, F], ValueError, r"Unalienable boolean indexer."),
        (
            slice("a"),
            TypeError,
            "slice indices must be integers or None or have an __index__ method",
        ),
    ],
)
def test__getitem__raises(container_or_child, key, err, msg):
    bc = container_or_child(a=0, b=0, c=0)
    with pytest.raises(err) as e:
        bc[key]  # noqa
    assert e.value.args[0].startswith(msg)


@pytest.mark.parametrize(
    "key,value,expected",
    [
        # list-like
        (["a"], ["a"], dict(a=0, b=None, c=None)),
        (pd.Index(["a"]), 0, dict(a=0, b=None, c=None)),
        (["a", "b"], [0, 0], dict(a=0, b=0, c=None)),
        (pd.Index(["a", "b"]), [0, 0], dict(a=0, b=0, c=None)),
        # slices
        (slice(None), [0, 0, 0], dict(a=0, b=0, c=0)),
        (slice(None, 10), [0, 0, 0], dict(a=0, b=0, c=0)),
        (slice(0, 2), [0, 0], dict(a=0, b=0, c=None)),  # exclusive right bound
        (slice(1, 1), [], dict(a=None, b=None, c=None)),  # exclusive right bound
        (slice(None, None, 2), [0, 0], dict(a=0, b=None, c=0)),
        # bool
        ([F, F, F], [], dict(a=None, b=None, c=None)),
        ([T, F, F], [0], dict(a=0, b=None, c=None)),
        ([T, F, T], [0, 0], dict(a=0, b=None, c=0)),
        ([T, T, T], [0, 0, 0], dict(a=0, b=0, c=0)),
    ],
)
def test__setitem__complex_keys(container_or_child, key, value, expected):
    result = container_or_child(a=None, b=None, c=None)
    result[key] = value

    expected = container_or_child(expected)
    assert result.keys() == expected.keys()
    assert result.index.equals(expected.index)
    assert list(result.values()) == list(expected.values())


@pytest.mark.parametrize(
    "key,value,expected",
    [
        # we allow scalar value for keys of length 1
        (["a"], 0, dict(a=0, b=None, c=None)),
        # list-like values
        (["a"], [0], dict(a=0, b=None, c=None)),
        (["a", "b"], pd.Index([0, 1]), dict(a=0, b=1, c=None)),
        # dict-like values
        (["a", "b"], dict(b=1, x=1), dict(a=1, b=1, c=None)),
        (["a", "b"], BaseContainer(b=1, x=1), dict(a=1, b=1, c=None)),
        (["a", "b"], AnyChild(b=1, x=1), dict(a=1, b=1, c=None)),
        # iterator - special treatment because it gets consumed
        (["a", "b"], "iterator-special", dict(a=0, b=1, c=None)),
    ],
)
def test__setitem__complex_keys(container_or_child, key, value, expected):
    if isinstance(value, str) and value == "iterator-special":
        value = (x for x in [0, 1])

    result = container_or_child(a=None, b=None, c=None)
    result[key] = value

    expected = container_or_child(expected)
    assert result.keys() == expected.keys()
    assert result.index.equals(expected.index)
    assert list(result.values()) == list(expected.values())


@pytest.mark.parametrize(
    "key,value,err,msg",
    [
        # bad keys
        ([T, F], None, ValueError, r"Unalienable boolean indexer."),
        ([T, F, F, F], None, ValueError, r"Unalienable boolean indexer."),
        (slice("a"), None, TypeError, "slice indices must be integers or None or"),
        # bad value-key combination
        ([T, F, T], 1, TypeError, r"value must be some kind of collection if"),
        ([T, F, T], [1, 2, 3], ValueError, r"Length mismatch: Got 2 keys, but"),
    ],
)
def test__setitem__raises(container_or_child, key, value, err, msg):
    bc = container_or_child(a=0, b=0, c=0)
    with pytest.raises(err) as e:
        bc[key] = value
    assert e.value.args[0].startswith(msg)
