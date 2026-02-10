from pathlib import Path
from typing import Literal, Optional, Tuple

from pydantic import BaseModel
from safetensors.torch import load_file, save_file

from .exceptions import ModaicError, RepositoryNotFoundError
from .hub import Commit, load_repo, sync_and_push
from .programs.predict import Predict

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - requires missing torch
    raise ImportError("torch must be installed to use ProbeModel") from exc


class ProbeConfig(BaseModel):
    probe_version: str = "v1"
    embedding_dim: int = 768
    model_path: str = "Qwen/Qwen3-VL-32B-Instruct"
    dropout: float = 0.0
    layer_index: int = -1  # Which layer was used for training (-1 means middle layer)
    num_layers: int | None = None  # Total number of layers in the source model (for reference)
    probe_type: Literal["linear", "nonlinear"] = "linear"  # Type of probe: "linear" or "nonlinear"


class ProbeModel(nn.Module):
    _source: Optional[Path] = None
    _source_commit: Optional[Commit] = None
    _is_probe: bool = True
    _from_auto: bool = False

    def __init__(self, config: ProbeConfig):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else None
        self.linear = nn.Linear(config.embedding_dim, 1)

    def forward(
        self,
        embeddings: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        if self.dropout is not None:
            embeddings = self.dropout(embeddings)
        logits = self.linear(embeddings)
        return torch.sigmoid(logits)

    @classmethod
    def load(cls, dir: Path | str) -> "ProbeModel":
        dir = Path(dir)

        if not (dir / "probe.safetensors").exists() or not (dir / "probe.json").exists():
            raise ModaicError(f"No probe model found in {dir}")

        with open(dir / "probe.json", "r") as f:
            config = ProbeConfig.model_validate_json(f.read())
        model = cls(config)
        state_dict = load_file(str(dir / "probe.safetensors"))
        model.load_state_dict(state_dict)
        model._source = dir
        return model

    def save(self, dir: Path | str):
        dir = Path(dir)
        save_file(self.state_dict(), str(dir / "probe.safetensors"))
        with open(dir / "probe.json", "w") as f:
            f.write(self.config.model_dump_json())

    def push_to_hub(
        self,
        repo: str,
        access_token: Optional[str] = None,
        commit_message: str = "(no commit message)",
        private: bool = False,
        branch: str = "main",
        tag: str = None,
    ) -> Commit:
        """
        Pushes the probe model to the given repo. Can be used on non-existing repos, existing repos with probes, or existing repos without probes.
        """
        sync_and_push(
            self,
            repo_path=repo,
            access_token=access_token,
            commit_message=commit_message,
            private=private,
            branch=branch,
            tag=tag,
            with_code=False,
        )


def load_probe(repo: str, access_token: Optional[str] = None, rev: str = "main") -> ProbeModel:
    local_dir, source_commit = load_repo(repo, access_token, rev=rev)

    model = ProbeModel.load(local_dir)
    model._source_commit = source_commit
    return model


if __name__ == "__main__":
    probe = ProbeModel(ProbeConfig(embedding_dim=777, layer_index=1))
    # probe.push_to_hub("test/probe", branch="main")
    # probe.push_to_hub("tytodd/new-probe", branch="main")
    probe.push_to_hub("tytodd/predict-test-repo", branch="main")
