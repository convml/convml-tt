# Copyright (c) 2008,2015,2018,2019 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Collection of generally useful utility code from the cookbook."""


class Registry:
    """Provide a generic function registry.
    This provides a class to instantiate, which then has a `register` method that can
    be used as a decorator on functions to register them under a particular name.
    """

    def __init__(self):
        """Initialize an empty registry."""
        self._registry = {}

    def register(self, name):
        """Register a callable with the registry under a particular name.
        Parameters
        ----------
        name : str
            The name under which to register a function
        Returns
        -------
        dec : callable
            A decorator that takes a function and will register it under the name.
        """

        def dec(func):
            self._registry[name] = func
            return func

        return dec

    def __getitem__(self, name):
        """Return any callable registered under name."""
        return self._registry[name]
