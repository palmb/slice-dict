#!/usr/bin/env python

from collections import UserList


class SimpleIndex(UserList):
    def difference(self, other):
        # set(self)-set(other) does not preserve order !
        return SimpleIndex(k for k in self if k not in other)

    @property
    def empty(self):
        return len(self) == 0

    def tolist(self):
        return self.data


try:
    from pandas import Index
except ImportError:
    Index = SimpleIndex
