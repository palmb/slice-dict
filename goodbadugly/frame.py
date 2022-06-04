#!/usr/bin/env python

from __future__ import annotations
import pandas as pd
from pandas._typing import Axes, Dtype
from pandas.api.types import is_list_like, is_hashable


class Frame:
    def __init__(
        self,
        data: dict = None,
        index: Axes = None,
        dtype: Dtype | None = None,
        copy: bool | None = None,
    ):
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise TypeError(f"only dicts are allowed")
        self._data = pd.Series(index=data.keys(), dtype="O")

        def to_series(o):
            return pd.Series(o, index=index, dtype=dtype, copy=copy)

        def to_frame(o):
            return pd.DataFrame(o, index=index, dtype=dtype, copy=copy)

        for key, value in data.items():
            if isinstance(value, pd.Series):
                value = to_series(value)
                value.name = None
            elif isinstance(value, pd.DataFrame):
                value = to_frame(value)
            else:
                try:
                    value = to_series(value)
                except Exception:
                    try:
                        value = to_frame(value)
                    except Exception:
                        raise TypeError(
                            f"{key=}. Cannot cast item of type {type(value)} to Series nor DataFrame"
                        ) from None

            self._data[key] = value

    @property
    def columns(self):
        return self._data.index

    @property
    def empty(self):
        return not len(self._data.index)

    def __len__(self):
        return len(self.columns)

    def __delitem__(self, key):
        if key in self.columns:
            del self._data[key]
        raise KeyError(key)

    def __getitem__(self, key) -> pd.Series | pd.DataFrame | Frame:
        if key in self.columns:
            return self._data[key]
        raise KeyError(key)

    def __setitem__(self, key, value):
        if not isinstance(value, (pd.Series, pd.DataFrame)):
            raise TypeError(type(value))
        self._data[key] = value

    def __str__(self):
        sep = " | "
        kws = dict(max_rows=30, min_rows=10)
        strings = {}
        for key, val in self._data.items():
            strings[key] = val.to_string(**kws).splitlines()
        header = ""
        vertline = ""
        for key, lines in strings.items():
            key = str(key)
            header += key.rjust(len(lines[0])) + sep
            vertline += "=" * max(len(lines[0]), len(key)) + sep

        final = "\n".join([header, vertline]) + "\n"
        i = 0
        while True:
            line = ""
            lines_left = len(strings.keys())
            for key, lines in strings.items():
                n = max(len(lines[0]), len(str(key)))
                if i >= len(lines):
                    part = " "
                    lines_left -= 1
                else:
                    part = lines[i]
                line += part.rjust(n) + sep
            if not lines_left:
                break
            final += line + "\n"
            i += 1

        return final



if __name__ == "__main__":

    fr = Frame(
        dict(
            a=pd.Series([1, 223, 4]),
            c=pd.Series([1, 223, 4]),
            b=pd.DataFrame(dict(a=[1, 2], b=[3, 45])),
            jasnkasjncak=pd.Series([1]),
            g=pd.Series([1, 223, 4]),
            jasnkk=pd.Series(index=pd.date_range("2020", "2021", 100)),
        )
    )

    print(fr)
    print(fr["c"])
