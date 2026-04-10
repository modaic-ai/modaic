import dspy
import pytest

import modaic


# ---------------------------------------------------------------------------
# Fixture — rotates through all three adapters
# ---------------------------------------------------------------------------


@pytest.fixture(
    params=[dspy.ChatAdapter, dspy.XMLAdapter, dspy.JSONAdapter],
    ids=["chat", "xml", "json"],
)
def adapter_setup(request):
    return request.param()


# ---------------------------------------------------------------------------
# Signatures
# ---------------------------------------------------------------------------


class ChoiceSig(dspy.Signature):
    """Answer the question."""

    question: str = dspy.InputField()
    answer: modaic.Enum["YES", "NO", "MAYBE"] = dspy.OutputField()


class SingleLetterSig(dspy.Signature):
    """Pick a letter."""

    question: str = dspy.InputField()
    choice: modaic.Enum["A", "B", "C", "D"] = dspy.OutputField()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def run_predict(adapter, noisy_value: str, *, field: str = "answer", sig=ChoiceSig) -> str:
    lm = dspy.utils.DummyLM([{field: noisy_value}], adapter=adapter)
    with dspy.context(lm=lm, adapter=adapter):
        result = dspy.Predict(sig)(question="test")
    return getattr(result, field)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_capital_normalization(adapter_setup):
    adapter = adapter_setup
    assert run_predict(adapter, "yes") == "YES"
    assert run_predict(adapter, "no") == "NO"
    assert run_predict(adapter, "maybe") == "MAYBE"
    assert run_predict(adapter, "Yes") == "YES"
    assert run_predict(adapter, "mAyBe") == "MAYBE"
    assert run_predict(adapter, "NO") == "NO"


def test_single_char_repeat(adapter_setup):
    adapter = adapter_setup
    assert run_predict(adapter, "AAAA", field="choice", sig=SingleLetterSig) == "A"
    assert run_predict(adapter, "bbbb", field="choice", sig=SingleLetterSig) == "B"
    assert run_predict(adapter, "CCC", field="choice", sig=SingleLetterSig) == "C"
    assert run_predict(adapter, "dd", field="choice", sig=SingleLetterSig) == "D"
    # Lowercase repeated → case-insensitive match
    assert run_predict(adapter, "aaaa", field="choice", sig=SingleLetterSig) == "A"


def test_parenthesis_wrap(adapter_setup):
    adapter = adapter_setup
    assert run_predict(adapter, "(YES)") == "YES"
    assert run_predict(adapter, "(NO)") == "NO"
    assert run_predict(adapter, "(MAYBE)") == "MAYBE"
    # Combined: parens + lowercase
    assert run_predict(adapter, "(yes)") == "YES"
    # Square brackets
    assert run_predict(adapter, "[YES]") == "YES"
    assert run_predict(adapter, "[no]") == "NO"
    assert run_predict(adapter, "[Maybe]") == "MAYBE"


def test_trailing_period(adapter_setup):
    adapter = adapter_setup
    assert run_predict(adapter, "YES.") == "YES"
    assert run_predict(adapter, "NO.") == "NO"
    assert run_predict(adapter, "MAYBE.") == "MAYBE"
    # Combined: trailing dot + lowercase
    assert run_predict(adapter, "yes.") == "YES"
    assert run_predict(adapter, "maybe.") == "MAYBE"


def test_edge_cases(adapter_setup):
    adapter = adapter_setup
    # Leading and trailing whitespace
    assert run_predict(adapter, "  YES  ") == "YES"
    assert run_predict(adapter, " maybe ") == "MAYBE"
    # Both-sided dots
    assert run_predict(adapter, ".YES.") == "YES"
    # Repeated single char + lowercase → requires both dedup + case normalization
    assert run_predict(adapter, "aaaa", field="choice", sig=SingleLetterSig) == "A"
    assert run_predict(adapter, "BBBB", field="choice", sig=SingleLetterSig) == "B"
    # Exact match passes through untouched
    assert run_predict(adapter, "YES") == "YES"
    assert run_predict(adapter, "A", field="choice", sig=SingleLetterSig) == "A"
    # Invalid value raises
    with pytest.raises(Exception):
        run_predict(adapter, "INVALID")
    with pytest.raises(Exception):
        run_predict(adapter, "ZZZZ", field="choice", sig=SingleLetterSig)
