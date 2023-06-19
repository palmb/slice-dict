#!/usr/bin/env python
from __future__ import annotations
import pickle
import pytest
from sliceable_dict import SliceDict, TypedSliceDict


@pytest.fixture(params=[SliceDict, TypedSliceDict])
def klass(request):
    return request.param


@pytest.mark.parametrize("kwargs", [dict(), dict(a=99), dict(x=99)])
@pytest.mark.parametrize("args", [(), ([[1, 2]],), (dict(a="a", b="b"),)])
def test_pickling(klass, args, kwargs):
    inst = klass(*args, **kwargs)
    result = pickle.loads(pickle.dumps(inst))
    assert isinstance(result, SliceDict)
    assert isinstance(result, klass)
    assert inst.keys() == result.keys()
    for k in inst.keys():
        assert result[k] == inst[k]
