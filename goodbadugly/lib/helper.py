#!/usr/bin/env python

import functools
import logging


def log_call(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logging.debug(f"{func.__name__} was called")
        return func(*args, **kwargs)
    return wrapper
