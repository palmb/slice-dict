#!/usr/bin/env python

from collections import UserList


class SimpleIndex(UserList):
    def difference(self, other):
        return SimpleIndex(set(self).difference(set(other)))

    @property
    def empty(self):
        return len(self) == 0

    def tolist(self):
        return self.data


try:
    from pandas import Index
except ImportError:
    Index = SimpleIndex
