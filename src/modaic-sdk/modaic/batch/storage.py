from __future__ import annotations

from pathlib import Path

from modaic_client import settings


def ensure_batch_storage_dirs() -> tuple[Path, Path]:
    settings.ensure_modaic_cache()
    batch_dir = settings.batch_dir
    batch_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = batch_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return batch_dir, tmp_dir


def get_batch_duckdb_path(batch_id: str) -> Path:
    batch_dir, _ = ensure_batch_storage_dirs()
    return batch_dir / f"{batch_id}.duckdb"


def get_modal_batch_parquet_paths(batch_id: str) -> tuple[Path, Path]:
    _, tmp_dir = ensure_batch_storage_dirs()
    return tmp_dir / f"{batch_id}.input.parquet", tmp_dir / f"{batch_id}.output.parquet"
