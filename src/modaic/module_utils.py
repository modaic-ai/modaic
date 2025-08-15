import sys
import os
from pathlib import Path
from types import ModuleType
from typing import Dict
import importlib.util
import sysconfig
import tempfile
import shutil


def is_builtin(module_name: str) -> bool:
    """Check whether a module name refers to a built-in module.

    Params:
      module_name: The fully qualified module name.

    Returns:
      bool: True if the module is a Python built-in.
    """

    return module_name in sys.builtin_module_names


def is_stdlib(module_name: str) -> bool:
    """Check whether a module belongs to the Python standard library.

    Params:
      module_name: The fully qualified module name.

    Returns:
      bool: True if the module is part of the stdlib (including built-ins).
    """

    try:
        spec = importlib.util.find_spec(module_name)
    except ValueError:
        # Some modules (e.g., __main__) may have __spec__ = None
        return False
    except Exception:
        return False
    if not spec:
        return False
    if spec.origin == "built-in":
        return True
    origin = spec.origin or ""
    stdlib_dir = Path(sysconfig.get_paths()["stdlib"]).resolve()
    try:
        origin_path = Path(origin).resolve()
    except OSError:
        return False
    return stdlib_dir in origin_path.parents or origin_path == stdlib_dir


def is_builtin_or_frozen(mod: ModuleType) -> bool:
    """Check whether a module object is built-in or frozen.

    Params:
      mod: The module object.

    Returns:
      bool: True if the module is built-in or frozen.
    """

    spec = getattr(mod, "__spec__", None)
    origin = getattr(spec, "origin", None)
    name = getattr(mod, "__name__", None)
    return (name in sys.builtin_module_names) or (origin in ("built-in", "frozen"))


def get_internal_imports() -> Dict[str, ModuleType]:
    """Return only internal modules currently loaded in sys.modules.

    Internal modules are defined as those not installed in site/dist packages
    (covers virtualenv `.venv` cases as well).

    If the environment variable `EDITABLE_MODE` is set to "true" (case-insensitive),
    modules located under `src/modaic/` are also excluded.

    Params:
      None

    Returns:
      Dict[str, ModuleType]: Mapping of module names to module objects that are
      not located under any "site-packages" or "dist-packages" directory.
    """

    def _is_in_site_or_dist_packages(path: Path) -> bool:
        parts = {p.lower() for p in path.parts}
        return "site-packages" in parts or "dist-packages" in parts

    internal: Dict[str, ModuleType] = {}
    editable_mode = os.getenv("EDITABLE_MODE", "false").lower() == "true"
    seen: set[int] = set()
    for name, module in list(sys.modules.items()):
        if module is None:
            continue
        # Deduplicate alias modules by identity
        module_id = id(module)
        if module_id in seen:
            continue
        seen.add(module_id)

        # Exclude built-in or frozen modules upfront
        if is_builtin_or_frozen(module):
            continue

        module_file = getattr(module, "__file__", None)
        if not module_file:
            # Skip built-ins and namespace packages without a file
            continue
        try:
            module_path = Path(module_file).resolve()
        except OSError:
            continue

        # Exclude built-in and stdlib modules
        if is_builtin(name) or is_stdlib(name):
            continue
        # Exclude third-party packages installed in site/dist-packages
        if _is_in_site_or_dist_packages(module_path):
            continue
        # Exclude local package source when running in editable mode
        if editable_mode:
            posix_path = module_path.as_posix().lower()
            if "src/modaic" in posix_path:
                continue
        internal[name] = module

    return internal


def materialize_internal_imports() -> Path:
    """Copy internal modules into a temporary directory inside MODAIC_CACHE_DIR.

    This function discovers internal modules via `get_internal_imports` and
    materializes their source files into an isolated temporary directory under
    `MODAIC_CACHE_DIR` (or `~/.cache/modaic` if not set). The directory layout
    mirrors Python's module naming (e.g., `pkg.sub.mod` → `pkg/sub/mod.py`), and
    package ancestor directories are created with `__init__.py` to ensure
    intra-module imports work within that directory.

    Params:
      None

    Returns:
      Path: Filesystem path to the created temporary directory containing the
      copied internal modules.
    """

    cache_dir_env = os.getenv("MODAIC_CACHE_DIR")
    default_cache_dir = Path(os.path.expanduser("~")) / ".cache" / "modaic"
    cache_dir = (
        Path(cache_dir_env).expanduser().resolve()
        if cache_dir_env
        else default_cache_dir.resolve()
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    temp_dir_path = Path(
        tempfile.mkdtemp(prefix="internal_modules_", dir=str(cache_dir))
    )

    internal_imports = get_internal_imports()

    seen_files: set[Path] = set()

    def _ensure_package_inits(base_dir: Path, name_parts: list[str]) -> None:
        # Create ancestor package dirs with __init__.py so relative imports work
        current = base_dir
        for part in name_parts:
            current = current / part
            current.mkdir(parents=True, exist_ok=True)
            init_file = current / "__init__.py"
            if not init_file.exists():
                init_file.touch()

    for module_name, module in internal_imports.items():
        module_file = getattr(module, "__file__", None)
        if not module_file:
            continue
        try:
            src_path = Path(module_file).resolve()
        except OSError:
            continue

        # Only copy Python source files
        if src_path.suffix != ".py":
            continue
        if src_path in seen_files:
            continue
        seen_files.add(src_path)

        name_parts = module_name.split(".")

        # Determine destination path
        if src_path.name == "__init__.py":
            # Package: create full package path and place __init__.py
            _ensure_package_inits(temp_dir_path, name_parts)
            dest_path = temp_dir_path.joinpath(*name_parts) / "__init__.py"
        else:
            # Module: ensure parent package path exists with __init__.py for ancestors
            if len(name_parts) > 1:
                _ensure_package_inits(temp_dir_path, name_parts[:-1])
            else:
                temp_dir_path.mkdir(parents=True, exist_ok=True)
            dest_path = (
                temp_dir_path.joinpath(*name_parts[:-1]) / f"{name_parts[-1]}.py"
            )

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dest_path)

    readme_src = Path("README.md")
    if readme_src.exists():
        readme_dest = temp_dir_path / "README.md"
        shutil.copy2(readme_src, readme_dest)

    return temp_dir_path


def create_pyproject_toml(temp_dir_path: Path):
    pyproject_toml_src = Path("pyproject.toml")
    if pyproject_toml_src.exists():
        pyproject_toml_dest = temp_dir_path / "pyproject.toml"
        shutil.copy2(pyproject_toml_src, pyproject_toml_dest)

    return temp_dir_path


def create_modaic_temp_dir():
    """
    Create a temporary directory inside the Modaic cache. Containing everything that will be pushed to the hub.
    Contains:
    - All internal modules used to run the agent
    - The pyproject.toml
    - The README.md
    - config.json
    - dspy_config.json
    """
    temp_dir_path = materialize_internal_imports()
