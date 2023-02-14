#!/usr/bin/env python
import pytest
import numpy as np
from sliceable_dict import SliceDict

try:
    import pandas as pd
except ImportError:
    from sliceable_dict.core import _SimpleIndex

    class _X:  # noqa
        def __init__(self, *args, **kwargs):
            return

    class pd:  # noqa
        Series = type("Series", (_X,), {})
        DataFrame = type("DataFrame", (_X,), {})
        Index = _SimpleIndex


T, F = True, False


class AnyChild(SliceDict):
    pass


@pytest.fixture(params=[SliceDict, AnyChild])
def container_or_child(request):
    return request.param


@pytest.mark.parametrize("kwargs", [dict(), dict(a=99), dict(x=99)])
@pytest.mark.parametrize("args", [(), ([[1, 2]],), (dict(a="a", b="b"),)])
def test_creation(container_or_child, args, kwargs):
    bc = container_or_child(*args, **kwargs)
    assert isinstance(bc, container_or_child)
    assert isinstance(bc, SliceDict)
    assert isinstance(bc, dict)


@pytest.mark.parametrize("attr", dir(dict))
def test_attrs(container_or_child, attr):
    assert hasattr(container_or_child, attr)


@pytest.mark.parametrize(
    "value",
    [
        None,
        1,
        "string",
        ["0", 1, 2.0],
        {"key": "value"},
        SliceDict(x="x"),
        AnyChild(x="x"),
        pd.Series(index=[1, 2], dtype=float),
        pd.DataFrame(1, index=[1, 2], columns=["c0", "c1"]),
    ],
)
def test_values(container_or_child, value):
    bc = container_or_child(a=value)
    assert isinstance(bc["a"], type(value))
    assert bc["a"] is value


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
    assert isinstance(result, SliceDict)

    expected = container_or_child(expected)
    assert result.keys() == expected.keys()


@pytest.mark.parametrize(
    "key,err,msg",
    [
        ("x", KeyError, "x"),
        (["a", "y"], KeyError, "keys ['y'] does not exist"),
        (["x", "y"], KeyError, "keys ['x', 'y'] does not exist"),
        ([T, F], ValueError, r"Boolean indexer has wrong length"),
        ([T, F, F, F], ValueError, r"Boolean indexer has wrong length"),
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
        (["a", "b"], SliceDict(b=1, x=1), dict(a=1, b=1, c=None)),
        (["a", "b"], AnyChild(b=1, x=1), dict(a=1, b=1, c=None)),
    ],
)
def test__setitem__complex_keys_test_values(container_or_child, key, value, expected):
    result = container_or_child(a=None, b=None, c=None)
    result[key] = value

    expected = container_or_child(expected)
    assert result.keys() == expected.keys()
    assert list(result.values()) == list(expected.values())


@pytest.mark.parametrize(
    "key,value,err,msg",
    [
        # bad keys
        ([T, F], None, ValueError, r"Boolean indexer has wrong length"),
        ([T, F, F, F], None, ValueError, r"Boolean indexer has wrong length"),
        (slice("a"), None, TypeError, "slice indices must be integers or None or"),
        # bad value-key combination
        ([T, F, T], 1, TypeError, r"Value must be some kind of collection if"),
        ([T, F, T], [1, 2, 3], ValueError, r"Got 2 keys, but 3 values"),
    ],
)
def test__setitem__raises(container_or_child, key, value, err, msg):
    bc = container_or_child(a=0, b=0, c=0)
    with pytest.raises(err) as e:
        bc[key] = value
    assert e.value.args[0].startswith(msg)
