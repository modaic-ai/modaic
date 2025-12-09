import importlib.util
import os
import re
import shutil
import sys
import sysconfig
import warnings
from pathlib import Path
from types import ModuleType
from typing import Dict

import tomlkit as tomlk

from .utils import compute_cache_dir

MODAIC_CACHE = compute_cache_dir()
PROGRAMS_CACHE = Path(MODAIC_CACHE) / "programs"
EDITABLE_MODE = os.getenv("EDITABLE_MODE", "false").lower() == "true"
TEMP_DIR = Path(MODAIC_CACHE) / "temp"


def is_builtin(module_name: str) -> bool:
    """Check whether a module name refers to a built-in module.

    Args:
      module_name: The fully qualified module name.

    Returns:
      bool: True if the module is a Python built-in.
    """

    return module_name in sys.builtin_module_names


def is_stdlib(module_name: str) -> bool:
    """Check whether a module belongs to the Python standard library.

    Args:
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

    Args:
      mod: The module object.

    Returns:
      bool: True if the module is built-in or frozen.
    """

    spec = getattr(mod, "__spec__", None)
    origin = getattr(spec, "origin", None)
    name = getattr(mod, "__name__", None)
    return (name in sys.builtin_module_names) or (origin in ("built-in", "frozen"))


# FIXME: make faster. Currently takes ~.70 seconds
def get_internal_imports() -> Dict[str, ModuleType]:
    """Return only internal modules currently loaded in sys.modules.

    Internal modules are defined as those not installed in site/dist packages
    (covers virtualenv `.venv` cases as well).

    If the environment variable `EDITABLE_MODE` is set to "true" (case-insensitive),
    modules located under `src/modaic/` are also excluded.

    Args:
      None

    Returns:
      Dict[str, ModuleType]: Mapping of module names to module objects that are
      not located under any "site-packages" or "dist-packages" directory.
    """

    internal: Dict[str, ModuleType] = {}

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

        # edge case: local modaic package
        if name == "modaic" or "modaic." in name:
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
        if EDITABLE_MODE:
            posix_path = module_path.as_posix().lower()
            if "src/modaic" in posix_path:
                continue
        normalized_name = name

        internal[normalized_name] = module

    return internal


def resolve_project_root() -> Path:
    """
    Return the project root directory, must be a directory containing a pyproject.toml file.

    Raises:
        FileNotFoundError: If pyproject.toml is not found in the current directory.
    """
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


def copy_module_layout(base_dir: Path, name_parts: list[str]) -> None:
    """
    Create ancestor package directories and ensure each contains an __init__.py file.
    Example:
        Given a base_dir of "/tmp/modaic" and name_parts of ["program","indexer"],
        creates the following layout:
        | /tmp/modaic/
        |   | program/
        |   |   | __init__.py
        |   | indexer/
        |   |   | __init__.py
    """
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


def init_program_repo(repo_path: str, with_code: bool = True) -> Path:
    """Create a local repository staging directory for program modules and files, excluding ignored files and folders."""
    repo_dir = TEMP_DIR / repo_path
    shutil.rmtree(repo_dir, ignore_errors=True)
    repo_dir.mkdir(parents=True, exist_ok=False)

    project_root = resolve_project_root()

    internal_imports = get_internal_imports()
    ignored_paths = get_ignored_files()

    seen_files: set[Path] = set()

    # Common repository files to include
    common_files = ["README.md", "LICENSE", "CONTRIBUTING.md"]

    for file_name in common_files:
        file_src = Path(file_name)
        if file_src.exists() and not is_path_ignored(file_src, ignored_paths):
            file_dest = repo_dir / file_name
            shutil.copy2(file_src, file_dest)
        elif file_name == "README.md":
            # Only warn for README.md since it's essential
            warnings.warn(
                "README.md not found in current directory. Please add one when pushing to the hub.",
                stacklevel=4,
            )

    if not with_code:
        return repo_dir

    for _, module in internal_imports.items():
        module_file = Path(getattr(module, "__file__", None))
        if not module_file:
            continue
        try:
            src_path = module_file.resolve()
        except OSError:
            continue
        if src_path.suffix != ".py":
            continue
        if is_path_ignored(src_path, ignored_paths):
            continue
        if src_path in seen_files:
            continue
        seen_files.add(src_path)

        rel_path = module_file.relative_to(project_root)
        dest_path = repo_dir / rel_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dest_path)

        # Ensure __init__.py is copied over at every directory level
        src_init = project_root / rel_path.parent / "__init__.py"
        dest_init = dest_path.parent / "__init__.py"
        if src_init.exists() and not dest_init.exists():
            shutil.copy2(src_init, dest_init)

    for extra_file in get_extra_files():
        if extra_file.is_dir():
            shutil.copytree(extra_file, repo_dir / extra_file.relative_to(project_root))
        else:
            shutil.copy2(extra_file, repo_dir / extra_file.relative_to(project_root))

    return repo_dir


def create_program_repo(repo_path: str, with_code: bool = True) -> Path:
    """
    Args:
        repo_path: The path to the repository.
        with_code: Whether to include the code in the repository.
        branch: The branch to post it to.
        tag: The tag to give it.
    Create a temporary directory inside the Modaic cache. Containing everything that will be pushed to the hub. This function adds the following files:
    - All internal modules used to run the program
    - The pyproject.toml
    - The README.md
    """
    package_name = repo_path.split("/")[-1]
    repo_dir = init_program_repo(repo_path, with_code=with_code)
    if with_code:
        create_pyproject_toml(repo_dir, package_name)

    return repo_dir


def get_ignored_files() -> list[Path]:
    """Return a list of absolute Paths that should be excluded from staging."""
    project_root = resolve_project_root()
    pyproject_path = Path("pyproject.toml")
    doc = tomlk.parse(pyproject_path.read_text(encoding="utf-8"))

    # Safely get [tool.modaic.exclude]
    files = (
        doc.get("tool", {})  # [tool]
        .get("modaic", {})  # [tool.modaic]
        .get("exclude", {})  # [tool.modaic.exclude]
        .get("files", [])  # [tool.modaic.exclude] files = ["file1", "file2"]
    )

    excluded: list[Path] = []
    for entry in files:
        entry = Path(entry)
        if not entry.is_absolute():
            entry = project_root / entry
        if entry.exists():
            excluded.append(entry)
    return excluded


def get_extra_files() -> list[Path]:
    """Return a list of extra files that should be excluded from staging."""
    project_root = resolve_project_root()
    pyproject_path = Path("pyproject.toml")
    doc = tomlk.parse(pyproject_path.read_text(encoding="utf-8"))
    files = (
        doc.get("tool", {})  # [tool]
        .get("modaic", {})  # [tool.modaic]
        .get("include", {})  # [tool.modaic.include]
        .get("files", [])  # [tool.modaic.include] files = ["file1", "file2"]
    )
    included: list[Path] = []
    for entry in files:
        entry = Path(entry)
        if entry.is_absolute():
            try:
                entry = entry.resolve()
                entry.relative_to(project_root.resolve())
            except ValueError:
                warnings.warn(
                    f"{entry} will not be bundled because it is not inside the current working directory",
                    stacklevel=4,
                )
        else:
            entry = project_root / entry
        if entry.resolve().exists():
            included.append(entry)

    return included


def create_pyproject_toml(repo_dir: Path, package_name: str):
    """
    Create a new pyproject.toml for the bundled program in the temp directory.
    """
    old = Path("pyproject.toml").read_text(encoding="utf-8")
    new = repo_dir / "pyproject.toml"

    doc_old = tomlk.parse(old)
    doc_new = tomlk.document()

    if "project" not in doc_old:
        raise KeyError("No [project] table in old TOML")
    doc_new["project"] = doc_old["project"]
    doc_new["project"]["dependencies"] = get_final_dependencies(doc_old["project"]["dependencies"])
    if "tool" in doc_old and "uv" in doc_old["tool"] and "sources" in doc_old["tool"]["uv"]:
        doc_new["tool"] = {"uv": {"sources": doc_old["tool"]["uv"]["sources"]}}
        warn_if_local(doc_new["tool"]["uv"]["sources"])

    doc_new["project"]["name"] = package_name

    with open(new, "w") as fp:
        tomlk.dump(doc_new, fp)


def get_final_dependencies(dependencies: list[str]) -> list[str]:
    """
    Get the dependencies that should be included in the bundled program.
    Filters out "[tool.modaic.ignore] dependencies. Adds [tool.modaic.include] dependencies.
    """
    pyproject_path = Path("pyproject.toml")
    doc = tomlk.parse(pyproject_path.read_text(encoding="utf-8"))

    # Safely get [tool.modaic.exclude]
    exclude_deps = (
        doc.get("tool", {})  # [tool]
        .get("modaic", {})  # [tool.modaic]
        .get("exclude", {})  # [tool.modaic.exclude]
        .get("dependencies", [])  # [tool.modaic.exclude] dependencies = ["praw", "sagemaker"]
    )
    include_deps = (
        doc.get("tool", {})  # [tool]
        .get("modaic", {})  # [tool.modaic]
        .get("include", {})  # [tool.modaic.include]
        .get("dependencies", [])  # [tool.modaic.include] dependencies = ["praw", "sagemaker"]
    )

    if exclude_deps:
        pattern = re.compile(r"\b(" + "|".join(map(re.escape, exclude_deps)) + r")\b")
        dependencies = [pkg for pkg in dependencies if not pattern.search(pkg)]
    return dependencies + include_deps


def warn_if_local(sources: dict[str, dict]):
    """
    Warn if the program is bundled with a local package.
    """
    for source, config in sources.items():
        if "path" in config:
            warnings.warn(
                f"Bundling program with local package {source} installed from {config['path']}. This is not recommended.",
                stacklevel=5,
            )


def _module_path(instance: object) -> str:
    """
    Return a deterministic module path for the given instance.

    Args:
      instance: The object instance whose class path should be resolved.

    Returns:
      str: A fully qualified path in the form "<module>.<ClassName>". If the
      class' module is "__main__", use the file system to derive a stable
      module name: the parent directory name when the file is "__main__.py",
      otherwise the file stem.
    """
    from .precompiled import PrecompiledConfig

    cls = type(instance)
    if cls is PrecompiledConfig:
        return "modaic.PrecompiledConfig"

    module_name = cls.__module__
    module = sys.modules[module_name]
    file = Path(module.__file__)
    module_path = str(file.relative_to(resolve_project_root()).with_suffix(""))
    module_path = module_path.replace("/", ".")

    return f"{module_path}.{cls.__name__}"
