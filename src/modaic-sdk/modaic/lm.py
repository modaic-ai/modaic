"""Backwards-compat shim: `modaic.LM` is now `modaic.SafeLM`."""

from modaic.safe_lm import SafeLM as LM

__all__ = ["LM"]
