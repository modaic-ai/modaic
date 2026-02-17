"""Tests for probe model version control with modaic.Predict."""

import json
import os
from pathlib import Path

import dspy
import pytest
import torch
from modaic.hub import get_user_info
from modaic.probe import ProbeConfig, ProbeModel
from modaic.programs.predict import Predict, PredictConfig
from modaic.utils import aggresive_rmtree, smart_rmtree
from modaic_client import settings
from safetensors.torch import load_file, save_file

from tests.utils import delete_program_repo

MODAIC_TOKEN = os.getenv("MODAIC_TOKEN")


class SummarizeSignature(dspy.Signature):
    """Summarize the given text into a concise summary."""

    text: str = dspy.InputField(desc="The text to summarize")
    summary: str = dspy.OutputField(desc="A concise summary of the text")


def create_probe_model(embedding_dim: int = 768, layer_index: int = -1) -> ProbeModel:
    """Create a ProbeModel with specific configuration for testing."""
    config = ProbeConfig(
        embedding_dim=embedding_dim,
        layer_index=layer_index,
        dropout=0.1,
        probe_type="linear",
    )
    return ProbeModel(config)


def save_probe_to_dir(probe: ProbeModel, dir: Path):
    """Save a probe model to a directory (working around any bugs in ProbeModel.save)."""
    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)
    # Save the weights
    save_file(probe.state_dict(), str(dir / "probe.safetensors"))
    # Save the config as JSON
    with open(dir / "probe.json", "w") as f:
        f.write(probe.config.model_dump_json())


def load_probe_from_dir(dir: Path) -> ProbeModel:
    """Load a probe model from a directory."""
    dir = Path(dir)
    with open(dir / "probe.json", "r") as f:
        config = ProbeConfig.model_validate_json(f.read())
    model = ProbeModel(config)
    state_dict = load_file(str(dir / "probe.safetensors"))
    model.load_state_dict(state_dict)
    return model


@pytest.fixture
def clean_folder() -> Path:
    smart_rmtree("tests/artifacts/temp/test_probe", ignore_errors=True)
    os.makedirs("tests/artifacts/temp/test_probe")
    return Path("tests/artifacts/temp/test_probe")


@pytest.fixture
def clean_modaic_cache() -> Path:
    aggresive_rmtree(settings.modaic_cache)
    return settings.modaic_cache


@pytest.fixture
def hub_repo(clean_modaic_cache: Path) -> str:
    if not MODAIC_TOKEN:
        pytest.skip("Skipping because MODAIC_TOKEN is not set")

    username = get_user_info(MODAIC_TOKEN)["login"]
    # delete the repo
    delete_program_repo(username=username, program_name="probe-test-repo", ignore_errors=True)
    return f"{username}/probe-test-repo"


# ====================
# Local Tests
# ====================


def test_probe_save_load_local(clean_folder: Path):
    """Test that ProbeModel can be saved and loaded locally."""
    probe = create_probe_model(embedding_dim=512, layer_index=3)

    # Set some specific weights to verify they're preserved
    with torch.no_grad():
        probe.linear.weight.fill_(0.5)
        probe.linear.bias.fill_(0.1)

    save_probe_to_dir(probe, clean_folder)

    assert os.path.exists(clean_folder / "probe.safetensors")
    assert os.path.exists(clean_folder / "probe.json")

    # Load and verify
    loaded_probe = load_probe_from_dir(clean_folder)
    assert loaded_probe.config.embedding_dim == 512
    assert loaded_probe.config.layer_index == 3
    assert loaded_probe.config.dropout == 0.1
    assert loaded_probe.config.probe_type == "linear"

    # Verify weights
    assert torch.allclose(loaded_probe.linear.weight, torch.full_like(loaded_probe.linear.weight, 0.5))
    assert torch.allclose(loaded_probe.linear.bias, torch.full_like(loaded_probe.linear.bias, 0.1))


def test_predict_with_probe_save_local(clean_folder: Path):
    """Test that Predict with probe saves probe files locally."""
    config = PredictConfig(signature=SummarizeSignature)
    predict = Predict(config)
    probe = create_probe_model(embedding_dim=768)

    # Set specific weights
    with torch.no_grad():
        probe.linear.weight.fill_(0.42)

    predict.probe = probe
    predict.save_precompiled(clean_folder)

    # Verify all files exist
    assert os.path.exists(clean_folder / "config.json")
    assert os.path.exists(clean_folder / "program.json")
    assert os.path.exists(clean_folder / "probe.safetensors")
    assert os.path.exists(clean_folder / "probe.json")

    # Verify probe config
    with open(clean_folder / "probe.json", "r") as f:
        probe_config = json.load(f)
    assert probe_config["embedding_dim"] == 768

    # Verify probe weights can be loaded
    loaded_probe = load_probe_from_dir(clean_folder)
    assert torch.allclose(loaded_probe.linear.weight, torch.full_like(loaded_probe.linear.weight, 0.42))


def test_predict_without_probe_no_probe_files_local(clean_folder: Path):
    """Test that Predict without probe doesn't create probe files."""
    config = PredictConfig(signature=SummarizeSignature)
    predict = Predict(config)
    predict.save_precompiled(clean_folder)

    assert os.path.exists(clean_folder / "config.json")
    assert os.path.exists(clean_folder / "program.json")
    assert not os.path.exists(clean_folder / "probe.safetensors")
    assert not os.path.exists(clean_folder / "probe.json")


def test_predict_copies_probe_from_source(clean_folder: Path):
    """Test that Predict copies probe files from source when probe is not explicitly set."""
    source_dir = clean_folder / "source"
    target_dir = clean_folder / "target"
    source_dir.mkdir(parents=True)
    target_dir.mkdir(parents=True)

    # Create initial predict with probe and save to source
    config = PredictConfig(signature=SummarizeSignature)
    predict1 = Predict(config)
    probe = create_probe_model(embedding_dim=512)
    with torch.no_grad():
        probe.linear.weight.fill_(0.99)
    predict1.probe = probe
    predict1.save_precompiled(source_dir)

    # Create new predict, set _source to source_dir, and save without setting probe
    predict2 = Predict(config)
    predict2._source = source_dir
    predict2.save_precompiled(target_dir)

    # Verify probe files were copied
    assert os.path.exists(target_dir / "probe.safetensors")
    assert os.path.exists(target_dir / "probe.json")

    # Verify probe config matches
    with open(target_dir / "probe.json", "r") as f:
        probe_config = json.load(f)
    assert probe_config["embedding_dim"] == 512

    # Verify weights were copied correctly
    loaded_probe = load_probe_from_dir(target_dir)
    assert torch.allclose(loaded_probe.linear.weight, torch.full_like(loaded_probe.linear.weight, 0.99))


# ====================
# Hub Tests
# ====================


def test_predict_push_with_probe(hub_repo: str):
    """Test that pushing Predict with probe uploads probe files to hub."""
    config = PredictConfig(signature=SummarizeSignature)
    predict = Predict(config)
    probe = create_probe_model(embedding_dim=768, layer_index=5)

    with torch.no_grad():
        probe.linear.weight.fill_(0.33)

    commit = predict.push_to_hub(hub_repo, probe=probe)
    staging_dir = settings.staging_dir / hub_repo

    assert commit is not None
    assert os.path.exists(staging_dir / "config.json")
    assert os.path.exists(staging_dir / "program.json")
    assert os.path.exists(staging_dir / "probe.safetensors")
    assert os.path.exists(staging_dir / "probe.json")

    # Verify probe config in staging
    with open(staging_dir / "probe.json", "r") as f:
        probe_config = json.load(f)
    assert probe_config["embedding_dim"] == 768
    assert probe_config["layer_index"] == 5


def test_predict_pull_with_probe(hub_repo: str):
    """Test that pulling Predict with probe downloads probe files from hub."""
    # Push with probe
    config = PredictConfig(signature=SummarizeSignature)
    predict = Predict(config)
    probe = create_probe_model(embedding_dim=1024, layer_index=10)

    with torch.no_grad():
        probe.linear.weight.fill_(0.77)
        probe.linear.bias.fill_(0.22)

    predict.push_to_hub(hub_repo, probe=probe)

    # Clear local cache
    repo_dir = Path(settings.modaic_hub_cache) / hub_repo
    smart_rmtree(repo_dir.parent, ignore_errors=True)

    # Pull from hub
    loaded_predict = Predict.from_precompiled(hub_repo)

    # Verify probe files exist in the loaded source
    assert loaded_predict._source is not None
    assert os.path.exists(loaded_predict._source / "probe.safetensors")
    assert os.path.exists(loaded_predict._source / "probe.json")

    # Load and verify probe
    loaded_probe = load_probe_from_dir(loaded_predict._source)
    assert loaded_probe.config.embedding_dim == 1024
    assert loaded_probe.config.layer_index == 10
    assert torch.allclose(loaded_probe.linear.weight, torch.full_like(loaded_probe.linear.weight, 0.77))
    assert torch.allclose(loaded_probe.linear.bias, torch.full_like(loaded_probe.linear.bias, 0.22))


def test_predict_push_without_probe_preserves_existing(hub_repo: str):
    """Test that pushing Predict without probe preserves existing probe from source."""
    # First push with probe
    config = PredictConfig(signature=SummarizeSignature)
    predict1 = Predict(config)
    probe = create_probe_model(embedding_dim=256, layer_index=2)

    with torch.no_grad():
        probe.linear.weight.fill_(0.55)

    predict1.push_to_hub(hub_repo, probe=probe, branch="main")
    print(predict1.dump_state())

    # Pull the predict
    loaded_predict = Predict.from_precompiled(hub_repo, rev="main", lm=dspy.LM("openai/gpt-4o-mini"))
    print(loaded_predict.dump_state())

    # Push again without specifying probe - should preserve existing probe
    loaded_predict.push_to_hub(hub_repo, branch="main", commit_message="update without probe")

    # Clear cache and pull again
    repo_dir = Path(settings.modaic_hub_cache) / hub_repo
    smart_rmtree(repo_dir.parent, ignore_errors=True)

    # Verify probe files still exist
    final_predict = Predict.from_precompiled(hub_repo, rev="main")
    assert os.path.exists(final_predict._source / "probe.safetensors")
    assert os.path.exists(final_predict._source / "probe.json")

    # Verify probe config and weights preserved
    loaded_probe = load_probe_from_dir(final_predict._source)
    assert loaded_probe.config.embedding_dim == 256
    assert loaded_probe.config.layer_index == 2
    assert torch.allclose(loaded_probe.linear.weight, torch.full_like(loaded_probe.linear.weight, 0.55))


def test_predict_probe_branching(hub_repo: str):
    """Test that different branches can have different probes."""
    config = PredictConfig(signature=SummarizeSignature)

    # Push to main with one probe
    predict_main = Predict(config)
    probe_main = create_probe_model(embedding_dim=768, layer_index=1)
    with torch.no_grad():
        probe_main.linear.weight.fill_(0.11)
    predict_main.push_to_hub(hub_repo, probe=probe_main, branch="main")

    # Load from main to get source info
    loaded_main = Predict.from_precompiled(hub_repo, rev="main")

    # Push to dev branch with different probe
    predict_dev = Predict(config)
    probe_dev = create_probe_model(embedding_dim=512, layer_index=3)
    with torch.no_grad():
        probe_dev.linear.weight.fill_(0.88)
    predict_dev._source = loaded_main._source
    predict_dev._source_commit = loaded_main._source_commit
    predict_dev.push_to_hub(hub_repo, probe=probe_dev, branch="dev")

    # Clear cache
    repo_cache = Path(settings.modaic_hub_cache) / hub_repo
    smart_rmtree(repo_cache.parent, ignore_errors=True)

    # Verify main branch has original probe
    main_predict = Predict.from_precompiled(hub_repo, rev="main")
    main_probe = load_probe_from_dir(main_predict._source)
    assert main_probe.config.embedding_dim == 768
    assert main_probe.config.layer_index == 1
    assert torch.allclose(main_probe.linear.weight, torch.full_like(main_probe.linear.weight, 0.11))

    # Verify dev branch has different probe
    dev_predict = Predict.from_precompiled(hub_repo, rev="dev")
    dev_probe = load_probe_from_dir(dev_predict._source)
    assert dev_probe.config.embedding_dim == 512
    assert dev_probe.config.layer_index == 3
    assert torch.allclose(dev_probe.linear.weight, torch.full_like(dev_probe.linear.weight, 0.88))


def test_predict_probe_with_tag(hub_repo: str):
    """Test that tagged versions preserve their probe configuration."""
    config = PredictConfig(signature=SummarizeSignature)

    # Push with probe and tag
    predict = Predict(config)
    probe = create_probe_model(embedding_dim=384, layer_index=6)
    with torch.no_grad():
        probe.linear.weight.fill_(0.44)
    predict.push_to_hub(hub_repo, probe=probe, branch="main", tag="v1.0.0")

    # Update main branch with different probe
    loaded = Predict.from_precompiled(hub_repo, rev="main")
    predict2 = Predict(config)
    probe2 = create_probe_model(embedding_dim=1024, layer_index=12)
    with torch.no_grad():
        probe2.linear.weight.fill_(0.99)
    predict2._source = loaded._source
    predict2._source_commit = loaded._source_commit
    predict2.push_to_hub(hub_repo, probe=probe2, branch="main")

    # Clear cache
    repo_cache = Path(settings.modaic_hub_cache) / hub_repo
    smart_rmtree(repo_cache.parent, ignore_errors=True)

    # Verify tag still has original probe
    tagged_predict = Predict.from_precompiled(hub_repo, rev="v1.0.0")
    tagged_probe = load_probe_from_dir(tagged_predict._source)
    assert tagged_probe.config.embedding_dim == 384
    assert tagged_probe.config.layer_index == 6
    assert torch.allclose(tagged_probe.linear.weight, torch.full_like(tagged_probe.linear.weight, 0.44))

    # Verify main has updated probe
    main_predict = Predict.from_precompiled(hub_repo, rev="main")
    main_probe = load_probe_from_dir(main_predict._source)
    assert main_probe.config.embedding_dim == 1024
    assert main_probe.config.layer_index == 12
    assert torch.allclose(main_probe.linear.weight, torch.full_like(main_probe.linear.weight, 0.99))


def test_predict_push_new_probe_replaces_existing(hub_repo: str):
    """Test that pushing with a new probe replaces the existing one."""
    config = PredictConfig(signature=SummarizeSignature)

    # Push with first probe
    predict1 = Predict(config)
    probe1 = create_probe_model(embedding_dim=768, layer_index=1)
    with torch.no_grad():
        probe1.linear.weight.fill_(0.11)
    predict1.push_to_hub(hub_repo, probe=probe1, branch="main")

    # Load and push with different probe
    loaded = Predict.from_precompiled(hub_repo, rev="main")
    probe2 = create_probe_model(embedding_dim=512, layer_index=5)
    with torch.no_grad():
        probe2.linear.weight.fill_(0.77)
    loaded.push_to_hub(hub_repo, probe=probe2, branch="main")

    # Clear cache and verify
    repo_cache = Path(settings.modaic_hub_cache) / hub_repo
    smart_rmtree(repo_cache.parent, ignore_errors=True)

    final_predict = Predict.from_precompiled(hub_repo, rev="main")
    final_probe = load_probe_from_dir(final_predict._source)

    # Should have the new probe config
    assert final_probe.config.embedding_dim == 512
    assert final_probe.config.layer_index == 5
    assert torch.allclose(final_probe.linear.weight, torch.full_like(final_probe.linear.weight, 0.77))


def test_predict_branch_without_probe_from_main_with_probe(hub_repo: str):
    """Test creating a branch from main that has probe, but new branch without probe."""
    config = PredictConfig(signature=SummarizeSignature)

    # Push main with probe
    predict_main = Predict(config)
    probe = create_probe_model(embedding_dim=768)
    with torch.no_grad():
        probe.linear.weight.fill_(0.66)
    predict_main.push_to_hub(hub_repo, probe=probe, branch="main")

    # Load from main
    loaded = Predict.from_precompiled(hub_repo, rev="main")

    # Create feature branch - explicitly set probe to None to not include it
    # Note: The current implementation will copy probe from _source if probe is None
    # This tests that behavior
    predict_feature = Predict(config, lm=dspy.LM("openai/gpt-4o-mini"))
    predict_feature._source = loaded._source
    predict_feature._source_commit = loaded._source_commit
    # Don't set probe - should copy from source
    predict_feature.push_to_hub(hub_repo, branch="feature-no-probe")

    # Clear cache
    repo_cache = Path(settings.modaic_hub_cache) / hub_repo
    smart_rmtree(repo_cache.parent, ignore_errors=True)

    # Feature branch should still have probe (copied from source)
    feature_predict = Predict.from_precompiled(hub_repo, rev="feature-no-probe")
    assert os.path.exists(feature_predict._source / "probe.safetensors")
    assert os.path.exists(feature_predict._source / "probe.json")

    feature_probe = load_probe_from_dir(feature_predict._source)
    assert feature_probe.config.embedding_dim == 768
    assert torch.allclose(feature_probe.linear.weight, torch.full_like(feature_probe.linear.weight, 0.66))
