#!/usr/bin/env python

from __future__ import annotations

import copy

import pandas as pd
from .base import _Axis, _BaseContainer, ColumnContainer


class Frame(ColumnContainer):
    _name = "Frame"

    def _set_single_item_callback(self, key, value):
        if not isinstance(value, (pd.Series, pd.DataFrame)):
            raise TypeError(f"Value must be of type pd.Series or "
                            f"pd.DataFrame, not {type(value)}")
        return super()._set_single_item_callback(key, value)

    @property
    def empty(self):
        return len(self.columns) == 0

    def __repr__(self):
        return str(self)

    def __str__(self):
        max_rows = pd.get_option('display.max_rows')
        min_rows = pd.get_option('display.min_rows')
        return self.to_string(max_rows=max_rows, min_rows=min_rows)

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
        if self.empty:
            return f"Empty {self._name}\nColumns: {self.columns.tolist()}"

        pd_to_string = dict(max_rows=max_rows, min_rows=min_rows)
        col_sep, head_sep = " | ", "="
        iterators = []

        def iter4ever(lines, key: str):
            n_chars = 0
            # make header: key and separating line
            for line in lines[:1]:
                n_chars = len(line)
                yield False, key.rjust(n_chars)
                yield False, head_sep * n_chars
            # skip column names if no df is present
            if not show_df_column_names:
                lines = lines[1:]
            # make body
            for line in lines:
                yield False, line
            # generate empty lines
            while True:
                yield True, " " * n_chars

        df_is_present = False
        for key, val in self.data.items():
            if val.empty:
                val = pd.Series("", index=[""])
            if isinstance(val, pd.Series):
                val = val.to_frame(name=" " * len(str(key)))
            else:
                df_is_present = True
            lines = val.to_string(**pd_to_string).splitlines()
            iterators.append(iter4ever(lines, str(key)))

        # If the user did not set `show_df_column_names` manually,
        # we show column names only if a dataframe is present
        # `show_df_column_names` is used in local function `iter4ever`
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


class SaqcFrame(Frame):
    def _set_single_item_callback(self, key, value):
        if not isinstance(key, str):
            raise TypeError(
                f"Keys must be of type string, not {type(key).__name__}"
            )
        if isinstance(value, list):
            value = pd.Series(value)
        if isinstance(value, pd.DataFrame):
            raise TypeError('saqc cant handle dataframes')

        return super()._set_single_item_callback(key, value)
