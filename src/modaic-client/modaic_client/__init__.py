from importlib import import_module

__all__ = ["Arbiter", "ModaicClient", "configure", "get_modaic_client", "settings", "track", "configure_modaic_client"]

_LAZY_IMPORTS = {
    "Arbiter": ("modaic_client.client", "Arbiter"),
    "ModaicClient": ("modaic_client.client", "ModaicClient"),
    "configure": ("modaic_client.config", "configure"),
    "configure_modaic_client": ("modaic_client.client", "configure_modaic_client"),
    "get_modaic_client": ("modaic_client.client", "get_modaic_client"),
    "settings": ("modaic_client.config", "settings"),
    "track": ("modaic_client.config", "track"),
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_IMPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
