import sys
import os
from pathlib import Path
from types import ModuleType
from typing import Dict
import importlib.util
import sysconfig
import tempfile
import shutil
import tomlkit as tomlk
import warnings
import re


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

    internal: Dict[str, ModuleType] = {}
    editable_mode = os.getenv("EDITABLE_MODE", "false").lower() == "true"
    seen: set[int] = set()
    for name, module in list(sys.modules.items()):
        if module is None:
            continue
        module_id = id(module)
        if module_id in seen:
            continue
        seen.add(module_id)

        if is_builtin_or_frozen(module):
            continue

        module_file = getattr(module, "__file__", None)
        if not module_file:
            continue
        try:
            module_path = Path(module_file).resolve()
        except OSError:
            continue

        if is_builtin(name) or is_stdlib(name):
            continue
        if is_external_package(module_path):
            continue
        if editable_mode:
            posix_path = module_path.as_posix().lower()
            if "src/modaic" in posix_path:
                continue
        internal[name] = module

    return internal


def compute_cache_dir() -> Path:
    """Return the cache directory used to stage internal modules."""
    cache_dir_env = os.getenv("MODAIC_CACHE_DIR")
    default_cache_dir = Path(os.path.expanduser("~")) / ".cache" / "modaic"
    cache_dir = (
        Path(cache_dir_env).expanduser().resolve()
        if cache_dir_env
        else default_cache_dir.resolve()
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def resolve_project_root() -> Path:
    """Return the project root directory, preferring the parent of pyproject.toml if present."""
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        raise FileNotFoundError("pyproject.toml not found in current directory")
    return pyproject_path.resolve().parent


def is_path_ignored(target_path: Path, ignored_paths: list[Path]) -> bool:
    """Return True if target_path matches or is contained within any ignored path."""
    try:
        absolute_target = target_path.resolve()
    except OSError:
        return False
    for ignored in ignored_paths:
        if absolute_target == ignored:
            return True
        try:
            absolute_target.relative_to(ignored)
            return True
        except Exception:
            pass
    return False


def ensure_package_initializers(base_dir: Path, name_parts: list[str]) -> None:
    """Create ancestor package directories and ensure each contains an __init__.py file."""
    current = base_dir
    for part in name_parts:
        current = current / part
        current.mkdir(parents=True, exist_ok=True)
        init_file = current / "__init__.py"
        if not init_file.exists():
            init_file.touch()


def is_external_package(path: Path) -> bool:
    """Return True if the path is under site-packages or dist-packages."""
    parts = {p.lower() for p in path.parts}
    return "site-packages" in parts or "dist-packages" in parts


def init_temp_dir() -> Path:
    """Create a temp directory and stage internal modules, excluding ignored files and folders."""
    cache_dir = compute_cache_dir()
    temp_dir_path = Path(
        tempfile.mkdtemp(prefix="internal_modules_", dir=str(cache_dir))
    )

    internal_imports = get_internal_imports()
    project_root = resolve_project_root()
    ignored_paths = get_ignored_files(project_root)

    seen_files: set[Path] = set()

    for module_name, module in internal_imports.items():
        module_file = getattr(module, "__file__", None)
        if not module_file:
            continue
        try:
            src_path = Path(module_file).resolve()
        except OSError:
            continue
        if src_path.suffix != ".py":
            continue
        if is_path_ignored(src_path, ignored_paths):
            continue
        if src_path in seen_files:
            continue
        seen_files.add(src_path)

        name_parts = module_name.split(".")
        if src_path.name == "__init__.py":
            ensure_package_initializers(temp_dir_path, name_parts)
            dest_path = temp_dir_path.joinpath(*name_parts) / "__init__.py"
        else:
            if len(name_parts) > 1:
                ensure_package_initializers(temp_dir_path, name_parts[:-1])
            else:
                temp_dir_path.mkdir(parents=True, exist_ok=True)
            dest_path = (
                temp_dir_path.joinpath(*name_parts[:-1]) / f"{name_parts[-1]}.py"
            )

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dest_path)

    readme_src = Path("README.md")
    if readme_src.exists() and not is_path_ignored(readme_src, ignored_paths):
        readme_dest = temp_dir_path / "README.md"
        shutil.copy2(readme_src, readme_dest)
    else:
        warnings.warn(
            "README.md not found in current directory. Please add one when pushing to the hub."
        )

    return temp_dir_path


def create_modaic_temp_dir(package_name: str) -> Path:
    """
    Create a temporary directory inside the Modaic cache. Containing everything that will be pushed to the hub. This function adds the following files:
    - All internal modules used to run the agent
    - The pyproject.toml
    - The README.md
    """
    temp_dir = init_temp_dir()
    print(f"Created temp pyproject.toml at {temp_dir}")
    create_pyproject_toml(temp_dir, package_name)

    return temp_dir


def get_ignored_files(project_root: Path) -> list[Path]:
    """Return a list of absolute Paths that should be excluded from staging."""
    pyproject_path = Path("pyproject.toml")
    doc = tomlk.parse(pyproject_path.read_text(encoding="utf-8"))

    # Safely get [tool.modaic.ignore]
    ignore_table = (
        doc.get("tool", {})  # [tool]
        .get("modaic", {})  # [tool.modaic]
        .get("ignore")  # [tool.modaic.ignore]
    )

    if ignore_table is None or "files" not in ignore_table:
        return []

    ignored: list[Path] = []
    for entry in ignore_table["files"]:
        try:
            ignored.append((project_root / entry).resolve())
        except OSError:
            continue
    return ignored


def create_pyproject_toml(temp_dir_path: Path, package_name: str):
    """
    Create a new pyproject.toml for the bundled agent in the temp directory.
    """
    old = Path("pyproject.toml").read_text(encoding="utf-8")
    new = temp_dir_path / "pyproject.toml"

    doc_old = tomlk.parse(old)
    doc_new = tomlk.document()
    # print(doc_old)

    if "project" not in doc_old:
        raise KeyError("No [project] table in old TOML")
    doc_new["project"] = doc_old["project"]
    doc_new["project"]["dependencies"] = get_filtered_dependencies(
        doc_old["project"]["dependencies"]
    )
    if (
        "tool" in doc_old
        and "uv" in doc_old["tool"]
        and "sources" in doc_old["tool"]["uv"]
    ):
        doc_new["tool"] = {"uv": {"sources": doc_old["tool"]["uv"]["sources"]}}
        warn_if_local(doc_new["tool"]["uv"]["sources"])

    doc_new["project"]["name"] = package_name

    with open(new, "w") as fp:
        tomlk.dump(doc_new, fp)


def get_filtered_dependencies(dependencies: list[str]) -> list[str]:
    """
    Get the dependencies that should be included in the bundled agent.
    """
    pyproject_path = Path("pyproject.toml")
    doc = tomlk.parse(pyproject_path.read_text(encoding="utf-8"))

    # Safely get [tool.modaic.ignore]
    ignore_table = (
        doc.get("tool", {})  # [tool]
        .get("modaic", {})  # [tool.modaic]
        .get("ignore", {})  # [tool.modaic.ignore]
    )

    if "dependencies" not in ignore_table:
        return dependencies

    ignored_dependencies = ignore_table["dependencies"]
    pattern = re.compile(
        r"\b(" + "|".join(map(re.escape, ignored_dependencies)) + r")\b"
    )
    filtered_dependencies = [pkg for pkg in dependencies if not pattern.search(pkg)]
    return filtered_dependencies


def warn_if_local(sources: list[dict]):
    """
    Warn if the agent is bundled with a local package.
    """
    for source, config in sources.items():
        print(source)
        if "path" in config:
            warnings.warn(
                f"Bundling agent with local package {source} installed from {config['path']}. This is not recommended."
            )
