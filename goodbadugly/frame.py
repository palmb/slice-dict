#!/usr/bin/env python

from __future__ import annotations
import pandas as pd
from pandas._typing import Axes, Dtype
from pandas.api.types import is_list_like, is_hashable, is_scalar


class Frame:
    _name = "Frame"

    def __init__(
        self,
        data: dict | list | Frame | None = None,
        columns: Axes | None = None,
        index: Axes | None = None,
        dtype: Dtype | None = None,
        copy: bool = False,
    ):
        if dtype is None:
            dtype = object

        if data is None or len(data) == 0:
            data = {}

        if isinstance(data, Frame):
            data = data._data.to_dict()

        if columns is not None:
            columns = pd.Index(columns)

        # handle non-empty lists and list-likes
        if not isinstance(data, dict):
            data = list(data)
            if columns is None:
                raise ValueError(f"'columns' must not be None, for list-like data")
            if len(data) != len(columns):
                raise ValueError(
                    f"{len(columns)} columns passed, "
                    f"but data imply {len(data)} columns."
                )
            data = dict(zip(columns, data))

        # now we must have a dict
        if not isinstance(data, dict):
            raise TypeError(f"Cannot create from data of type {type(data).__name__!r}")

        if columns is None:
            columns = pd.Index(data.keys())

        self._data = pd.Series(index=columns, dtype=object)

        for col in columns:

            if col not in data:
                continue

            value = data[col]

            if isinstance(value, (pd.Series, pd.DataFrame)):
                value = value.copy(deep=copy)
                value.name = None
            else:
                try:
                    value = pd.Series(value, index=index, dtype=dtype, copy=copy)
                except Exception:
                    try:
                        value = pd.DataFrame(value, index=index, dtype=dtype, copy=copy)
                    except Exception:
                        raise TypeError(
                            f"{col=}. Cannot cast item from type "
                            f"{type(value).__name__!r} to Series nor DataFrame"
                        ) from None

            self._data[col] = value

    @property
    def columns(self):
        return self._data.index

    @property
    def empty(self):
        return self._data.isna().all()

    def __len__(self):
        return len(self.columns)

    def __contains__(self, item):
        return item in self._data.index

    def __iter__(self):
        yield from self._data.index

    def __delitem__(self, key):
        if key in self.columns:
            del self._data[key]
        raise KeyError(key)

    def __getitem__(self, key) -> pd.Series | pd.DataFrame | Frame:
        if key in self.columns:
            return self._data[key]
        if isinstance(key, (list, pd.Index)):
            _data = {}
            for k in key:
                _data[k] = self[k]
            return Frame(_data, copy=False)
        raise KeyError(key)

    def __setitem__(self, key, value):
        if isinstance(value, (pd.Series, pd.DataFrame)):
            self._data[key] = value

        elif isinstance(value, Frame):
            if isinstance(key, (list, pd.Index)):
                if len(key) != len(value.columns):
                    raise ValueError(
                        f"key has {len(key)} columns, but value "
                        f"has {len(value.columns)} columns"
                    )
                for i, k in enumerate(key):
                    self._data[k] = value[value.columns[i]]
            elif is_hashable(key) and len(value.columns) == 1:
                self._data[key] = value[value.columns[0]]
            else:
                raise TypeError(
                    f"key must be of type list or pd.Index, not {type(key).__name__!r} "
                )
        else:
            raise TypeError(f"{type(value).__name__!r}")

    def __str__(self):
        return self.to_string(max_rows=30, min_rows=10)

    def to_string(self, max_rows=None, min_rows=None, show_df_column_names=None):
        """
        Render a Frame to a console-friendly tabular output.

        Parameters
        ----------
        max_rows : int, optional
            Maximum number of rows to display in the console.

        min_rows : int, optional
            The number of rows to display in the console in a
            truncated repr (when number of rows is above max_rows).

        show_df_column_names : bool, default None
            Prints the name of dataframe columns. If `None` the
            space reserved for column names is removed if
            not necessary (no Dataframes present).

        Returns
        -------
        str
            Returns the result as a string.
        """
        to_pandas_kws = dict(max_rows=max_rows, min_rows=min_rows)

        if self.empty:
            return f"Empty {self._name}\nColumns: {self.columns.tolist()}"

        col_sep, head_sep = " | ", "="
        iterators = []

        def iter4ever(lines, key: str):
            l = 0
            for i, line in enumerate(lines):
                l = len(line)
                if i == 0:
                    yield False, key.rjust(l)
                    yield False, head_sep * l
                    if not show_df_column_names:
                        continue
                yield False, line
            while True:
                yield True, " " * l

        df_is_present = False
        for key, val in self.data.items():
            if val.empty:
                val = pd.Series("", index=[""])

            if isinstance(val, pd.Series):
                val = val.to_frame(name=" "*len(str(key)))
            else:
                df_is_present = True

            lines = val.to_string(**to_pandas_kws).splitlines()
            iterators.append(iter4ever(lines, str(key)))

        # If the user did not set `show_df_column_names` manually,
        # we show column names only if a dataframe is present
        if show_df_column_names is None:
            show_df_column_names = df_is_present

        string = ""
        while True:
            exhausted = 0
            line = ""
            for iterator in iterators:
                i, s = next(iterator)
                line += s + col_sep
                exhausted += i
            if exhausted == len(iterators):
                break
            string += f"{line}\n"

        return string

    def __repr__(self):
        return str(self)


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

    fr2 = Frame(
        dict(
            a=pd.Series(0, index=range(10)),
            b=pd.Series(0, index=range(10)),
        )
    )

    # fr[list('b')] = fr2
    fr[list("ab")] = fr2
    # fr[:] = fr2

    print(fr)
    print(fr["c"])
