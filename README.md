# sliceable-dict

A simple dictionary that support slicing and multiple key selection.

## Installation and requirements

*Slice-dict* requires Python 3.7+ to run and can be installed by running 
```bash
pip install sliceable-dict
```

## Usage
One can use `SliceDict` identical to `dict`, but it brings some additional
features, like getting and setting multiple keys at once and slicing.

```pycon
>>> from sliceable_dict import SliceDict
>>> d = SliceDict(zero=0, one=1)
>>> d
{'zero': 0, 'one': 1,}

>>> d[['two', 'three']] = 2, 3
>>> d[1:-1]
{'one': 1, 'two': 2}
```
