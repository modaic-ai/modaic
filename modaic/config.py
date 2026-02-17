import sys
from pathlib import Path
from typing import Optional

from platformdirs import user_cache_dir
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_cache() -> str:
    if sys.platform.startswith("win"):
        return str(Path(user_cache_dir("modaic", appauthor=False)).resolve())
    return str((Path.home() / ".cache" / "modaic").resolve())


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    modaic_cache: str = Field(default_factory=_default_cache)
    modaic_token: Optional[str] = None
    modaic_git_url: str = "https://git.modaic.dev"
    modaic_api_url: str = "https://api.modaic.dev"
    editable_mode: bool = False
    track: bool = False

    @field_validator("modaic_git_url", "modaic_api_url")
    @classmethod
    def strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")

    @property
    def modaic_hub_cache(self) -> Path:
        return Path(self.modaic_cache) / "modaic_hub" / "modaic_hub"

    @property
    def staging_dir(self) -> Path:
        return Path(self.modaic_cache) / "staging"

    @property
    def sync_dir(self) -> Path:
        return Path(self.modaic_cache) / "sync"

    @property
    def batch_dir(self) -> Path:
        return Path(self.modaic_cache) / "batch"

    @property
    def use_github(self) -> bool:
        return "github.com" in self.modaic_git_url

    def ensure_modaic_cache(self) -> Path:
        path = Path(self.modaic_cache)
        path.mkdir(parents=True, exist_ok=True)
        return path


settings = Settings()


def configure(
    modaic_cache: Optional[str] = None,
    modaic_token: Optional[str] = None,
    modaic_git_url: Optional[str] = None,
    modaic_api_url: Optional[str] = None,
    track: Optional[bool] = None,
) -> None:
    if modaic_cache is not None:
        settings.modaic_cache = modaic_cache
    if modaic_token is not None:
        settings.modaic_token = modaic_token
    if modaic_git_url is not None:
        settings.modaic_git_url = modaic_git_url
    if modaic_api_url is not None:
        settings.modaic_api_url = modaic_api_url
    if track is not None:
        settings.track = track


def track() -> None:
    settings.track = True
