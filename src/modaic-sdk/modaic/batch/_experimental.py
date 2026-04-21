"""Mark a batch client class as experimental.

Instantiating an experimental client emits a :class:`UserWarning` so users
know the client is not covered by integration tests and the API may change.
"""

from __future__ import annotations

import warnings
from typing import Callable, TypeVar

T = TypeVar("T", bound=type)


def experimental(cls: T) -> T:
    cls._experimental = True  # type: ignore[attr-defined]
    orig_init: Callable = cls.__init__

    def _init(self, *args, **kwargs):
        warnings.warn(
            f"{cls.__name__} is experimental and not covered by tests. "
            "API may change; use with caution.",
            stacklevel=2,
        )
        orig_init(self, *args, **kwargs)

    cls.__init__ = _init  # type: ignore[method-assign]
    return cls
