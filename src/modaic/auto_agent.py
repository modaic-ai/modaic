# mini_auto.py
import json, importlib, pathlib, sys
from functools import lru_cache
import hashlib, os, subprocess, tempfile
from pathlib import Path

_REGISTRY = {}      # maps model_type string -> (ConfigCls, ModelCls)

def register(model_type: str, config_cls, model_cls):
    _REGISTRY[model_type] = (config_cls, model_cls)

@lru_cache
def _load_dynamic_class(repo_id, class_path):
    repo_dir = git_snapshot(repo_id)
    if repo_dir not in sys.path:               # make intra‑repo imports work
        sys.path.append(repo_dir)
    module_name, _, attr = class_path.rpartition(".")
    module = importlib.import_module(module_name)
    return getattr(module, attr)

class AutoConfig:
    @classmethod
    def from_precompiled(cls, repo_path):
        if os.path.exists(repo_path):
            cfg_path = os.path.join(repo_path, "config.json")
        else:
            cfg_path = git_snapshot(repo_path) + "/config.json"
        cfg = json.load(open(cfg_path))
        mt  = cfg["agent_type"]

        if mt in _REGISTRY:                    # built‑in
            C, _ = _REGISTRY[mt]
            return C.from_dict(cfg)

        # dynamic
        dyn_path = cfg["auto_map"]["AutoConfig"]
        DynConfig = _load_dynamic_class(repo_path, dyn_path)
        return DynConfig.from_dict(cfg)

class AutoAgent:
    @classmethod
    def from_precompiled(cls, repo_id, **kw):
        cfg_dir = git_snapshot(repo_id)
        cfg_path = git_snapshot(repo_id) + "/config.json"
        cfg_dict = json.load(open(cfg_path))
        cfg = AutoConfig.from_precompiled(cfg_dir)
        mt  = cfg.agent_type

        if mt in _REGISTRY:
            _, M = _REGISTRY[mt]
        else:
            dyn_path = cfg_dict["auto_map"]["AutoAgent"]
            M = _load_dynamic_class(repo_id, dyn_path)
            
        return M(config=cfg, **kw)


def git_snapshot(url: str, *, rev: str | None = "main") -> str:
    """
    Clone / update a public Git repo into a local cache and return the path.

    • url  – https://github.com/user/repo.git  •  git@server:repo.git  •  etc.
    • rev  – branch, tag, or full commit SHA; default is 'main'
    """
    # 1) pick a stable cache directory name based on url+rev
    cache_root = Path(os.getenv("MY_AUTO_CACHE", "~/.cache/my_auto")).expanduser()
    h = hashlib.sha1(f"{url}@{rev}".encode()).hexdigest()[:10]
    repo_dir = cache_root / h
    cache_root.mkdir(parents=True, exist_ok=True)

    if not repo_dir.exists():
        # first time → clone
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", rev, url, str(repo_dir)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    else:
        # already cached → fetch & checkout
        subprocess.run(["git", "-C", str(repo_dir), "fetch", "--depth", "1"], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(
            ["git", "-C", str(repo_dir), "reset", "--hard", f"origin/{rev}"],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    return str(repo_dir)
