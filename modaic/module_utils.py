import importlib.util
import re
import shutil
import sys
import sysconfig
import warnings
from pathlib import Path
from types import ModuleType
from typing import Dict

import tomlkit as tomlk

from .constants import EDITABLE_MODE, SYNC_DIR
from .utils import smart_rmtree


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


def get_extra_paths() -> list[Path]:
    """Return a list of extra files and folders that should be included in staging."""
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
    if sys.platform.startswith("win"):
        module_path = module_path.replace("\\", ".")
    else:
        module_path = module_path.replace("/", ".")

    return f"{module_path}.{cls.__name__}"


def create_sync_dir(repo_path: str, with_code: bool = True) -> Path:
    """Creates the 'sync' directory for the given repository path.
    - Contains a symlink directory layout of all files that will be pushed to modaic hub
    - The resulting directory is used to sync with a git repo in TEMP_DIR which orchestrates git operations
    """
    sync_dir = SYNC_DIR / repo_path
    smart_rmtree(sync_dir, ignore_errors=True)
    sync_dir.mkdir(parents=True, exist_ok=False)

    project_root = resolve_project_root()

    internal_imports = get_internal_imports()
    ignored_paths = get_ignored_files()

    seen_files: set[Path] = set()

    # Common repository files to include
    common_files = ["README.md", "LICENSE", "CONTRIBUTING.md"]

    for file_name in common_files:
        file_src = project_root / file_name
        if file_src.exists() and not is_path_ignored(file_src, ignored_paths):
            sync_file = sync_dir / file_name
            smart_link(sync_file, file_src)
        elif file_name == "README.md":
            # Only warn for README.md since it's essential
            warnings.warn(
                "README.md not found in current directory. Please add one when pushing to the hub.",
                stacklevel=4,
            )

    if not with_code:
        return sync_dir

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
        sync_path = sync_dir / rel_path
        sync_path.parent.mkdir(parents=True, exist_ok=True)
        smart_link(sync_path, src_path)

        # Ensure __init__.py is copied over at every directory level
        src_init = project_root / rel_path.parent / "__init__.py"
        sync_init = sync_path.parent / "__init__.py"
        if src_init.exists() and not sync_init.exists():
            smart_link(sync_init, src_init)
            seen_files.add(src_init)

    for extra_file in get_extra_paths():
        sync_path = sync_dir / extra_file.relative_to(project_root)
        smart_link(sync_path, extra_file)

    package_name = repo_path.split("/")[-1]
    create_pyproject_toml(sync_dir, package_name)

    return sync_dir


def sync_dir_from(source_dir: Path) -> Path:
    """Mirror the source directory as symlinks to a new directory."""
    # Expects directory from programs dir. programs/user/repo/rev
    # Make target directory  sync/user/repo
    sync_dir = SYNC_DIR / source_dir.parent.parent.name / source_dir.parent.name
    smart_rmtree(sync_dir, ignore_errors=True)
    sync_dir.mkdir(parents=True, exist_ok=False)
    excluded_names = {".git", "program.json", "config.json"}

    for src_path in source_dir.iterdir():
        if src_path.name in excluded_names:
            continue
        sync_path = sync_dir / src_path.relative_to(source_dir)
        smart_link(sync_path, src_path)

    return sync_dir


def smart_link(link: Path, source: Path) -> None:
    """
    If on mac/linux use symlink
    If on windows use hardlink for files and recursive hardlink for directories
    """
    if sys.platform.startswith("win"):
        if source.is_dir():
            link.parent.mkdir(parents=True, exist_ok=True)
            recursive_hard_link(link, source)
        else:
            link.hardlink_to(source)
    else:
        link.symlink_to(source, target_is_directory=source.is_dir())


def recursive_hard_link(link: Path, source: Path) -> None:
    """
    Create a hard link to the source directory.
    """
    if source.is_dir():
        link.mkdir(parents=True, exist_ok=True)
        for src_path in source.iterdir():
            recursive_hard_link(link / src_path.name, src_path)

    else:
        link.hardlink_to(source)


def _clear_git_repo(repo_dir: Path) -> None:
    """
    Clear the git repository of all files and directories except .git.
    """
    for path in repo_dir.iterdir():
        if path != repo_dir / ".git":
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()


# not in use currently
def copy_update_program_dir(target_dir: Path, repo_path: str, with_code: bool = True) -> None:
    """
    Copys files from workspace to the staging directory. (Used for Windows)
    Args:
        target_dir: The directory to copy the files to.
        repo_path: The path to the repository on modaic hub
        with_code: Whether to copy the code files.
    """
    _clear_git_repo(target_dir)
    project_root = resolve_project_root()

    internal_imports = get_internal_imports()
    ignored_paths = get_ignored_files()

    seen_files: set[Path] = set()

    # Common repository files to include
    common_files = ["README.md", "LICENSE", "CONTRIBUTING.md"]
    keep = set()
    for file_name in common_files:
        file_src = project_root / file_name
        if file_src.exists() and not is_path_ignored(file_src, ignored_paths):
            target_file = target_dir / file_name
            shutil.copy2(file_src, target_file)
            keep.add(target_file)
        elif file_name == "README.md":
            # Only warn for README.md since it's essential
            warnings.warn(
                "README.md not found in current directory. Please add one when pushing to the hub.",
                stacklevel=4,
            )

    if not with_code:
        return

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
        target_path = target_dir / rel_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, target_path)
        keep.add(target_path)

        # Ensure __init__.py is copied over at every directory level
        src_init = project_root / rel_path.parent / "__init__.py"
        target_init = target_path.parent / "__init__.py"
        if src_init.exists() and not target_init.exists():
            shutil.copy2(src_init, target_init)
            keep.add(target_init)
            seen_files.add(src_init)

    for extra_path in get_extra_paths():
        target_path = target_dir / extra_path.relative_to(project_root)
        if extra_path.is_dir():
            shutil.copytree(extra_path, target_path)
        else:
            shutil.copy2(extra_path, target_path)
        keep.add(target_path)

    package_name = repo_path.split("/")[-1]
    create_pyproject_toml(target_dir, package_name)


# Not in use currently
def copy_update_from(target_dir: Path, source_dir: Path) -> None:
    """
    Update target dir by copying in files from source directory.
    """
    _clear_git_repo(target_dir)
    for src_path in source_dir.iterdir():
        if src_path != source_dir / ".git":
            if src_path.is_dir():
                shutil.copytree(src_path, target_dir / src_path.name)
            else:
                shutil.copy2(src_path, target_dir / src_path.name)
