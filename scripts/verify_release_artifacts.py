#!/usr/bin/env python3
"""Verify that built Python distributions contain the intended release version."""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from email.parser import BytesParser
from email.policy import default
from pathlib import Path


def _metadata_version(data: bytes, artifact: Path) -> str:
    metadata = BytesParser(policy=default).parsebytes(data)
    if metadata["Name"] != "modaic":
        raise ValueError(f"{artifact.name}: expected package name modaic, got {metadata['Name']!r}")
    version = metadata["Version"]
    if version is None:
        raise ValueError(f"{artifact.name}: package metadata has no Version field")
    return version


def _wheel_version(artifact: Path) -> str:
    with zipfile.ZipFile(artifact) as archive:
        metadata_files = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_files) != 1:
            raise ValueError(
                f"{artifact.name}: expected one .dist-info/METADATA file, "
                f"found {len(metadata_files)}"
            )
        return _metadata_version(archive.read(metadata_files[0]), artifact)


def _sdist_version(artifact: Path) -> str:
    with tarfile.open(artifact, mode="r:gz") as archive:
        metadata_files = [
            member
            for member in archive.getmembers()
            if member.name.count("/") == 1 and member.name.endswith("/PKG-INFO")
        ]
        if len(metadata_files) != 1:
            raise ValueError(
                f"{artifact.name}: expected one top-level PKG-INFO file, "
                f"found {len(metadata_files)}"
            )
        extracted = archive.extractfile(metadata_files[0])
        if extracted is None:
            raise ValueError(f"{artifact.name}: could not read PKG-INFO")
        return _metadata_version(extracted.read(), artifact)


def verify(expected_version: str, dist_dir: Path) -> None:
    wheels = sorted(dist_dir.glob("*.whl"))
    sdists = sorted(dist_dir.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError(
            f"expected one wheel and one source distribution in {dist_dir}, "
            f"found {len(wheels)} wheel(s) and {len(sdists)} source distribution(s)"
        )

    for artifact, actual_version in (
        (wheels[0], _wheel_version(wheels[0])),
        (sdists[0], _sdist_version(sdists[0])),
    ):
        if actual_version != expected_version:
            raise ValueError(
                f"{artifact.name}: expected version {expected_version}, got {actual_version}"
            )
        print(f"Verified {artifact.name}: modaic {actual_version}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("expected_version")
    parser.add_argument("dist_dir", type=Path)
    args = parser.parse_args()
    verify(args.expected_version, args.dist_dir)


if __name__ == "__main__":
    main()
