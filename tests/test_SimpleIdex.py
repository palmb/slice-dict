#!/usr/bin/env python
from __future__ import annotations

import pytest
from sliceable_dict.core import _SimpleIndex


class MinimalListLike:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]


@pytest.mark.parametrize(
    "value",
    [
        # empty
        [],
        (),
        {},
        #
        [1, 2],
        MinimalListLike([1, 2]),
    ],
)
def test_construct(value):
    result = _SimpleIndex(value)
    assert isinstance(result, _SimpleIndex)
    assert result == list(value)


@pytest.mark.parametrize("name", [n for n in dir(list) if not n.startswith("__")])
def test_list_behavior(name):
    assert hasattr(_SimpleIndex, name)
    inst = getattr(_SimpleIndex([0, 1, 2]), name)
    exp = getattr(list([0, 1, 2]), name)

    # with no arg
    try:
        expected = exp()
    except Exception as e:
        with pytest.raises(type(e)):
            result = inst()
    else:
        result = inst()
        assert result == expected

    # with arg
    i = 1
    try:
        expected = exp(i)
    except Exception as e:
        with pytest.raises(type(e)):
            result = inst(i)
    else:
        result = inst(i)
        assert result == expected
