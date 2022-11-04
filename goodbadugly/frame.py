#!/usr/bin/env python

from __future__ import annotations

import copy

import pandas as pd
from .base import ColumnContainer


class Frame(ColumnContainer):
    _name = "Frame"

    def __init__(self, dict_=None, **kwargs):
        super().__init__(dict_, **kwargs)

    def _single_value_callback(self, value):
        if not isinstance(value, (pd.Series, pd.DataFrame)):
            raise TypeError(
                f"value must be of type pd.Series or pd.DataFrame, not {type(value)}"
            )
        return value

    def _prepare_value(self, value):
        return copy.copy(value)

    @property
    def empty(self):
        return len(self.columns) == 0

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
                val = val.to_frame(name=" " * len(str(key)))
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
