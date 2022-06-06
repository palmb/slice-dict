#!/usr/bin/env python
import numpy as np
import pytest
import pandas as pd
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
    assert isinstance(bc, dict)


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


def test_index_setter(container_or_child):
    bc = container_or_child(a=10, b=20, c=30)
    bc.index = [1, 2, 3]
    assert bc.keys() == dict.fromkeys([1, 2, 3]).keys()
    with pytest.raises(ValueError):
        bc.index = ['a', 'b']

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
        # slices
        (slice(None), dict(a=0, b=0, c=0)),
        (slice(0, 3), dict(a=0, b=0, c=0)),
        (slice(1, 1), dict()),
        (slice(None, 10), dict(a=0, b=0, c=0)),
        (slice(None, None, 2), dict(a=0, c=0)),
        # bool
        ([F, F, F], dict()),
        ([T, F, F], dict(a=0)),
        ([T, F, T], dict(a=0, c=0)),
        ([T, T, T], dict(a=0, b=0, c=0)),
    ],
)
def test__getitem___complex_key(container_or_child, key, expected):
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
def test__getitem___raises(container_or_child, key, err, msg):
    bc = container_or_child(a=0, b=0, c=0)
    with pytest.raises(err) as e:
        bc[key]  # noqa
    assert e.value.args[0].startswith(msg)
