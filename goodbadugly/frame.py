#!/usr/bin/env python

from __future__ import annotations

import warnings

from typing import List, ValuesView, Iterable, Tuple, Any, Hashable

import abc
import numpy as np
import functools

import pandas as pd
from .base import _Axis, _BaseContainer
from .formatting import Formatter


class IndexMixin:
    @abc.abstractmethod
    def values(self) -> ValuesView[pd.Series | pd.DataFrame]:
        ...

    def __get_indexes(self):
        indexes = []
        for obj in self.values():
            indexes.append(obj.index)
        return indexes

    def _union_index(self):
        return functools.reduce(pd.Index.union, self.__get_indexes(), pd.Index([]))

    def _shared_index(self):
        indexes = self.__get_indexes()
        if indexes:
            return functools.reduce(pd.Index.intersection, indexes)
        return pd.Index([])


class Frame(_BaseContainer, IndexMixin):
    columns = _Axis("columns")

    @property
    def _constructor(self) -> type[Frame]:
        return Frame

    def _set_single_item_callback(self, key, value):
        if not isinstance(value, (pd.Series, pd.DataFrame)):
            raise TypeError(
                f"Value must be of type pd.Series or "
                f"pd.DataFrame, not {type(value)}"
            )
        return super()._set_single_item_callback(key, value)

    @property
    def empty(self):
        return len(self.columns) == 0

    def _uniquify_key(self, name, postfix="_new"):
        if name not in self.keys():
            return name
        name += postfix
        if name not in self.keys():
            return name
        i = 1
        while f"{name}{i}" in self.keys():
            i += 1
        return f"{name}{i}"

    def flatten(self):
        """
        Promote dataframe columns to first level columns.

        Prepend column names of an inner dataframe with the
        key/column name of the outer frame.

        Examples
        --------
        >>> frame = Frame(key0=pd.DataFrame(np.arange(4).reshape(2,2), columns=['c0', 'c1']))
        >>> frame
             key0 |
        ========= |
           c0  c1 |
        0   0   1 |
        1   2   3 |

        >>> frame.flatten()
           key0_c0 |    key0_c1 |
        ========== | ========== |
        0        0 | 0        1 |
        1        2 | 1        3 |
        """
        data = {}
        for key, value in self.items():
            if isinstance(value, pd.DataFrame):
                for col, ser in dict(value).items():
                    data[self._uniquify_key(f"{key}_{col}")] = ser
            else:
                data[key] = value
        return self.__class__(data)

    def to_dataframe(self):
        return pd.DataFrame(dict(self.flatten()))

    def __repr__(self):
        return str(self)

    def __str__(self):
        max_rows = pd.get_option("display.max_rows")
        min_rows = pd.get_option("display.min_rows")
        return self.to_string(max_rows=max_rows, min_rows=min_rows)

    def to_string(
        self,
        max_rows: int | None = None,
        min_rows: int | None = None,
        show_df_column_names: bool = True,
    ):
        """
        Render a Frame to a console-friendly tabular output.

        Parameters
        ----------
        max_rows : int, optional
            Maximum number of rows to display in the console.

        min_rows : int, optional
            The number of rows to display in the console in a
            truncated repr (when number of rows is above max_rows).

        show_df_column_names : bool, default True
            Prints column names of dataframes if True,
            otherwise colum names are hidden.

        Returns
        -------
        str
            Returns the result as a string.
        """
        return Formatter(self, max_rows, min_rows, show_df_column_names).to_string()



class SaqcFrame(Frame):  # dios replacement
    def _set_single_item_callback(self, key, value):
        if not isinstance(key, str):
            raise TypeError(f"Keys must be of type string, not {type(key).__name__}")
        if isinstance(value, list):
            value = pd.Series(value)
        # if isinstance(value, pd.DataFrame):
        #     raise TypeError('saqc cant handle dataframes')

        return super()._set_single_item_callback(key, value)

    def to_df(self):
        warnings.warn(
            "to_df() is deprecated, use " "to_dataframe() instead.", DeprecationWarning
        )
        return self.to_dataframe()
