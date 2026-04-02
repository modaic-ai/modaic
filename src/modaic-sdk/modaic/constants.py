"""Deprecated: use ``from modaic import settings`` instead.

This module re-exports legacy constant names so that external packages
that depend on ``modaic.constants`` continue to work.  Every access
emits a :class:`DeprecationWarning`.
"""

import warnings
from pathlib import Path


def __getattr__(name: str):
    from modaic_client import settings

    _MAPPING = {  # noqa: N806
        "MODAIC_CACHE": lambda: Path(settings.modaic_cache),
        "MODAIC_HUB_CACHE": lambda: Path(settings.modaic_hub_cache),
        "EDITABLE_MODE": lambda: settings.editable_mode,
        "STAGING_DIR": lambda: Path(settings.staging_dir),
        "SYNC_DIR": lambda: Path(settings.sync_dir),
        "MODAIC_TOKEN": lambda: settings.modaic_token,
        "MODAIC_GIT_URL": lambda: settings.modaic_git_url,
        "USE_GITHUB": lambda: settings.use_github,
        "MODAIC_API_URL": lambda: settings.modaic_api_url,
        "BATCH_DIR": lambda: Path(settings.batch_dir),
    }

    if name in _MAPPING:
        warnings.warn(
            f"modaic.constants.{name} is deprecated. "
            f"Use `from modaic import settings` and access `settings.{name.lower()}` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _MAPPING[name]()

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
