import modulefinder
import sysconfig
import pathlib
from typing import Dict

project_root = pathlib.Path(__file__).parent.resolve()
stdlib_dir = pathlib.Path(sysconfig.get_paths()["stdlib"]).resolve()


def is_internal(path: str | pathlib.Path | None) -> bool:
    """Return True if the given path belongs to the project tree.

    Params:
      path: File system path or None.
    """
    if not path:
        return False
    try:
        p = pathlib.Path(path).resolve()
    except OSError:
        return False
    return project_root in p.parents or p == project_root


def is_stdlib(path: str | pathlib.Path | None) -> bool:
    """Return True if the given path belongs to the Python standard library.

    Params:
      path: File system path or None.
    """
    if not path:
        return False
    try:
        p = pathlib.Path(path).resolve()
    except OSError:
        return False
    return stdlib_dir in p.parents or p == stdlib_dir


def is_external(path: str | pathlib.Path | None) -> bool:
    """Return True for non-internal, non-stdlib modules (i.e., third-party).

    Params:
      path: File system path or None.
    """
    if not path:
        return False
    return not is_internal(path) and not is_stdlib(path)


class InternalOnlyFinder(modulefinder.ModuleFinder):
    """Module finder that only recurses into imports originating from internal modules."""

    def scan_code(self, co, m):
        """Skip scanning into external modules to avoid recursion beyond project code.

        Params:
          co: Code object being scanned.
          m: Module object that owns the code object.
        """
        mod_file = getattr(m, "__file__", None) if m is not None else None
        if mod_file and not is_internal(mod_file):
            return
        return super().scan_code(co, m)


finder = InternalOnlyFinder()
script_path = project_root / "my_module1.py"
finder.run_script(str(script_path))

internal_modules: Dict[str, pathlib.Path] = {}
external_modules: Dict[str, pathlib.Path] = {}

for name, mod in finder.modules.items():
    mod_file = getattr(mod, "__file__", None)
    if not mod_file:
        # Skip built-ins and modules without a file
        continue
    path = pathlib.Path(mod_file).resolve()
    if is_internal(path):
        internal_modules[name] = path
    elif is_external(path):
        external_modules[name] = path

for name in sorted(internal_modules.keys()):
    print(name, internal_modules[name])

for name in sorted(external_modules.keys()):
    print(name, external_modules[name])
