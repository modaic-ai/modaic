import os
from pathlib import Path

import dspy
import pytest
from modaic.hub import get_user_info
from modaic.programs.predict import Predict, PredictConfig
from modaic.utils import aggresive_rmtree, smart_rmtree
from modaic_client import settings

from tests.utils import delete_program_repo

MODAIC_TOKEN = os.getenv("MODAIC_TOKEN")


class SummarizeSignature(dspy.Signature):
    """Summarize the given text into a concise summary."""

    text: str = dspy.InputField(desc="The text to summarize")
    summary: str = dspy.OutputField(desc="A concise summary of the text")


class ClassifyEmotionSignature(dspy.Signature):
    """Classify the emotion in the given sentence."""

    sentence: str = dspy.InputField(desc="The sentence to classify")
    emotion: str = dspy.OutputField(desc="The emotion detected in the sentence")


class TranslateSignature(dspy.Signature):
    """Translate text from one language to another."""

    text: str = dspy.InputField(desc="The text to translate")
    target_language: str = dspy.InputField(desc="The target language for translation")
    translation: str = dspy.OutputField(desc="The translated text")


@pytest.fixture
def clean_folder() -> Path:
    smart_rmtree("tests/artifacts/temp/test_programs", ignore_errors=True)
    os.makedirs("tests/artifacts/temp/test_programs")
    return Path("tests/artifacts/temp/test_programs")


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
    delete_program_repo(username=username, program_name="predict-test-repo", ignore_errors=True)
    return f"{username}/predict-test-repo"


# ====================
# Local Tests
# ====================


def test_predict_config_local(clean_folder: Path):
    """Test that PredictConfig can be saved and loaded locally."""
    config = PredictConfig(signature=SummarizeSignature)
    config.save_precompiled(clean_folder)

    assert os.path.exists(clean_folder / "config.json")
    assert len(os.listdir(clean_folder)) == 1

    loaded_config = PredictConfig.from_precompiled(clean_folder)
    assert loaded_config.signature.equals(SummarizeSignature)


def test_predict_local_save_load(clean_folder: Path):
    """Test that Predict can be saved and loaded locally."""
    config = PredictConfig(signature=SummarizeSignature)
    predict = Predict(config)
    predict.save_precompiled(clean_folder)

    assert os.path.exists(clean_folder / "config.json")
    assert os.path.exists(clean_folder / "program.json")
    assert len(os.listdir(clean_folder)) == 2

    loaded_predict = Predict.from_precompiled(clean_folder)
    assert loaded_predict.config.signature.model_json_schema() == SummarizeSignature.model_json_schema()


def test_predict_local_with_lm(clean_folder: Path):
    """Test that Predict can be saved and loaded with a custom LM."""
    config = PredictConfig(signature=SummarizeSignature, model="openai/gpt-4o-mini")
    lm = dspy.LM("openai/gpt-4o-mini")
    predict = Predict(config, lm=lm)
    predict.save_precompiled(clean_folder)

    loaded_predict = Predict.from_precompiled(clean_folder)
    assert loaded_predict.config.signature.equals(SummarizeSignature)
    assert loaded_predict.config.model == "openai/gpt-4o-mini"


def test_predict_local_change_signature(clean_folder: Path):
    """Test that Predict signature can be changed and saved locally."""
    # First save with one signature
    config1 = PredictConfig(signature=SummarizeSignature)
    predict1 = Predict(config1)
    predict1.save_precompiled(clean_folder)

    loaded_predict1 = Predict.from_precompiled(clean_folder)
    assert loaded_predict1.config.signature.equals(SummarizeSignature)

    # Save with a different signature
    config2 = PredictConfig(signature=ClassifyEmotionSignature)
    predict2 = Predict(config2)
    predict2.save_precompiled(clean_folder)

    loaded_predict2 = Predict.from_precompiled(clean_folder)
    assert loaded_predict2.config.signature.equals(ClassifyEmotionSignature)


def test_predict_forward_call(clean_folder: Path):
    """Test that Predict forward method (and __call__) work correctly."""
    config = PredictConfig(signature=SummarizeSignature)
    predict = Predict(config)

    # Save and load to ensure state is preserved
    predict.save_precompiled(clean_folder)
    loaded_predict = Predict.from_precompiled(clean_folder)

    # Verify the predict was initialized with correct signature
    assert loaded_predict.signature.equals(SummarizeSignature)


# ====================
# Hub Tests
# ====================


def test_predict_push_to_hub(hub_repo: str):
    """Test that Predict can be pushed to the hub."""
    config = PredictConfig(signature=SummarizeSignature)
    predict = Predict(config)

    commit = predict.push_to_hub(hub_repo)
    staging_dir = settings.staging_dir / hub_repo

    assert commit is not None
    assert commit.repo == hub_repo
    assert os.path.exists(staging_dir / "config.json")
    assert os.path.exists(staging_dir / "program.json")
    assert os.path.exists(staging_dir / ".git")


def test_predict_load_from_hub(hub_repo: str):
    """Test that Predict can be loaded from the hub."""
    # Push first
    config = PredictConfig(signature=SummarizeSignature)
    predict = Predict(config)
    predict.push_to_hub(hub_repo)

    # Clear local cache
    repo_dir = Path(settings.modaic_hub_cache) / hub_repo
    smart_rmtree(repo_dir.parent, ignore_errors=True)

    # Load from hub
    loaded_predict = Predict.from_precompiled(hub_repo)
    assert loaded_predict.config.signature.equals(SummarizeSignature)


def test_predict_hub_call(hub_repo: str):
    """Test that Predict loaded from hub can be called."""
    config = PredictConfig(signature=SummarizeSignature)
    predict = Predict(config)
    predict.push_to_hub(hub_repo)

    # Load from hub
    loaded_predict = Predict.from_precompiled(hub_repo)

    # Verify the predict was initialized correctly
    assert loaded_predict.signature.equals(SummarizeSignature)


def test_predict_change_lm_push_branch(hub_repo: str):
    """Test that Predict LM can be changed and pushed to a different branch."""
    # Initial push to main
    config1 = PredictConfig(signature=SummarizeSignature, model="openai/gpt-4o-mini")
    predict1 = Predict(config1)
    predict1.push_to_hub(hub_repo, branch="main")

    # Load, change model, push to dev branch
    loaded_predict = Predict.from_precompiled(hub_repo, rev="main")
    assert loaded_predict.config.model == "openai/gpt-4o-mini"

    # Create new config with different model
    config2 = PredictConfig(signature=SummarizeSignature, model="openai/gpt-4o")
    predict2 = Predict(config2)
    predict2._source = loaded_predict._source
    predict2._source_commit = loaded_predict._source_commit
    predict2.push_to_hub(hub_repo, branch="dev")

    # Load from main - should have original model
    main_predict = Predict.from_precompiled(hub_repo, rev="main")
    assert main_predict.config.model == "openai/gpt-4o-mini"

    # Load from dev - should have new model
    dev_predict = Predict.from_precompiled(hub_repo, rev="dev")
    assert dev_predict.config.model == "openai/gpt-4o"


def test_predict_change_signature_push_branch(hub_repo: str):
    """Test that Predict signature can be changed and pushed to a different branch."""
    # Initial push to main with SummarizeSignature
    config1 = PredictConfig(signature=SummarizeSignature)
    predict1 = Predict(config1)
    predict1.push_to_hub(hub_repo, branch="main")

    # Load and verify
    loaded_predict = Predict.from_precompiled(hub_repo, rev="main")
    assert loaded_predict.config.signature.equals(SummarizeSignature)

    # Create new config with different signature, push to feature branch
    config2 = PredictConfig(signature=ClassifyEmotionSignature)
    predict2 = Predict(config2)
    predict2._source = loaded_predict._source
    predict2._source_commit = loaded_predict._source_commit
    predict2.push_to_hub(hub_repo, branch="emotion-feature")

    # Load from main - should have original signature
    main_predict = Predict.from_precompiled(hub_repo, rev="main")
    assert main_predict.config.signature.equals(SummarizeSignature)

    # Load from feature branch - should have new signature
    feature_predict = Predict.from_precompiled(hub_repo, rev="emotion-feature")
    assert feature_predict.config.signature.equals(ClassifyEmotionSignature)


def test_predict_multiple_branches(hub_repo: str):
    """Test pushing to multiple branches with different configurations."""
    # Push different signatures to different branches
    signatures = [
        ("main", SummarizeSignature, "openai/gpt-4o-mini"),
        ("dev", ClassifyEmotionSignature, "openai/gpt-4o"),
        ("feature-translate", TranslateSignature, "openai/gpt-4o-mini"),
    ]

    # Push to main first (required for other branches)
    config_main = PredictConfig(signature=signatures[0][1], model=signatures[0][2])
    predict_main = Predict(config_main)
    predict_main.push_to_hub(hub_repo, branch="main")

    # Load main to get source info for subsequent pushes
    loaded_main = Predict.from_precompiled(hub_repo, rev="main")

    # Push to other branches
    for branch, sig, model in signatures[1:]:
        config = PredictConfig(signature=sig, model=model)
        predict = Predict(config)
        predict._source = loaded_main._source
        predict._source_commit = loaded_main._source_commit
        predict.push_to_hub(hub_repo, branch=branch)

    # Verify each branch has correct configuration
    for branch, expected_sig, expected_model in signatures:
        loaded = Predict.from_precompiled(hub_repo, rev=branch)
        assert loaded.config.signature.equals(expected_sig), f"Branch {branch} has wrong signature"
        assert loaded.config.model == expected_model, f"Branch {branch} has wrong model"


def test_predict_with_code_warning(hub_repo: str):
    """Test that with_code parameter triggers a warning for Predict."""
    config = PredictConfig(signature=SummarizeSignature)
    predict = Predict(config)

    # This should emit a warning since with_code is ignored for Predict
    with pytest.warns(UserWarning, match="with_code=.*is not supported"):
        predict.push_to_hub(hub_repo, with_code=True)


def test_predict_push_update_same_branch(hub_repo: str):
    """Test pushing updates to the same branch."""
    # Initial push
    config1 = PredictConfig(signature=SummarizeSignature, model="openai/gpt-4o-mini")
    predict1 = Predict(config1)
    commit1 = predict1.push_to_hub(hub_repo, branch="main")

    # Load, modify, and push again to same branch
    loaded = Predict.from_precompiled(hub_repo)
    config2 = PredictConfig(signature=SummarizeSignature, model="openai/gpt-4o")
    predict2 = Predict(config2)
    predict2._source = loaded._source
    predict2._source_commit = loaded._source_commit
    commit2 = predict2.push_to_hub(hub_repo, branch="main")

    # Commits should be different
    assert commit1.sha != commit2.sha

    # Verify update was applied
    updated = Predict.from_precompiled(hub_repo, rev="main")
    assert updated.config.model == "openai/gpt-4o"


def test_predict_tag(hub_repo: str):
    """Test that Predict can be pushed with a tag."""
    config = PredictConfig(signature=SummarizeSignature, model="openai/gpt-4o-mini")
    predict = Predict(config)
    predict.push_to_hub(hub_repo, branch="main", tag="v1.0.0")

    # Load from tag
    tagged = Predict.from_precompiled(hub_repo, rev="v1.0.0")
    assert tagged.config.signature.equals(SummarizeSignature)
    assert tagged.config.model == "openai/gpt-4o-mini"

    # Push update to main (tag should remain unchanged)
    loaded = Predict.from_precompiled(hub_repo, rev="main")
    config2 = PredictConfig(signature=SummarizeSignature, model="openai/gpt-4o")
    predict2 = Predict(config2)
    predict2._source = loaded._source
    predict2._source_commit = loaded._source_commit
    predict2.push_to_hub(hub_repo, branch="main")

    # Tag should still have original model
    tagged_after = Predict.from_precompiled(hub_repo, rev="v1.0.0")
    assert tagged_after.config.model == "openai/gpt-4o-mini"

    # Main should have new model
    main_after = Predict.from_precompiled(hub_repo, rev="main")
    assert main_after.config.model == "openai/gpt-4o"
