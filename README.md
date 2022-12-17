# sliceable-dict

A simple dictionary that support slicing and multiple key selection.

## Installation and requirements

*sliceable-dict* requires Python 3.7+ to run and can be installed by running 
```bash
pip install sliceable-dict
```

## Usage
The provided class `SliceDict` is an extended dictionary. It behaves exactly as the 
buildin `dict` for single (hashable) keys, but adds some additional features. 
Namely, getting and setting multiple keys at once, slicing with integers and boolean
selection.

An example usage:
```pycon
>>> from sliceable_dict import SliceDict
>>> d = SliceDict(zero=0, one=1)
>>> d
{'zero': 0, 'one': 1,}

>>> isinstance(d, dict)
True

# multi-key support 
>>> d[['two', 'three']] = 2, 3
>>> d[['one', 'three']]
{'one': 1, 'three': 3}

# slicing
>>> d[:]
{'zero': 0, 'one': 1, 'two': 2, 'three': 3}
>>> d[1:-1]
{'one': 1, 'two': 2}
>>> d[1::2]
{'one': 1, 'three': 3}

# boolean selection
>>> bool_list = [True, False, True, False]
>>> selection = d[bool_list]
>>> selection
{'zero': 0, 'two': 2}

# views as for buildin dict
>>> kv = selection.keys()
>>> kv
KeysView({'zero': 0, 'two': 2})
>>> list(kv)
['zero', 'two']

>>> vv = selection.values()
>>> vv
ValuesView({'zero': 0, 'two': 2})
>>> list(vv)
[0, 2]
```
