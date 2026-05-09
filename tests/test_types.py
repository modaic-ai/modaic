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


# ---------------------------------------------------------------------------
# Scale
# ---------------------------------------------------------------------------


class RatingSig(dspy.Signature):
    """Rate the thing."""

    question: str = dspy.InputField()
    rating: modaic.Scale[1, 5] = dspy.OutputField()


def test_scale_int_passthrough(adapter_setup):
    adapter = adapter_setup
    assert run_predict(adapter, 3, field="rating", sig=RatingSig) == 3
    assert run_predict(adapter, 1, field="rating", sig=RatingSig) == 1
    assert run_predict(adapter, 5, field="rating", sig=RatingSig) == 5


def test_scale_string_coercion(adapter_setup):
    adapter = adapter_setup
    assert run_predict(adapter, "3", field="rating", sig=RatingSig) == 3
    assert run_predict(adapter, " 4 ", field="rating", sig=RatingSig) == 4
    assert run_predict(adapter, "3.", field="rating", sig=RatingSig) == 3
    assert run_predict(adapter, ".2.", field="rating", sig=RatingSig) == 2


def test_scale_bracket_wrap(adapter_setup):
    adapter = adapter_setup
    assert run_predict(adapter, "(4)", field="rating", sig=RatingSig) == 4
    assert run_predict(adapter, "[2]", field="rating", sig=RatingSig) == 2


def test_scale_float_coercion(adapter_setup):
    adapter = adapter_setup
    assert run_predict(adapter, "3.0", field="rating", sig=RatingSig) == 3
    assert run_predict(adapter, "5.0", field="rating", sig=RatingSig) == 5


def test_scale_out_of_range_raises(adapter_setup):
    adapter = adapter_setup
    with pytest.raises(Exception):
        run_predict(adapter, "0", field="rating", sig=RatingSig)
    with pytest.raises(Exception):
        run_predict(adapter, "6", field="rating", sig=RatingSig)
    with pytest.raises(Exception):
        run_predict(adapter, "-1", field="rating", sig=RatingSig)


def test_scale_non_numeric_raises(adapter_setup):
    adapter = adapter_setup
    with pytest.raises(Exception):
        run_predict(adapter, "abc", field="rating", sig=RatingSig)
    with pytest.raises(Exception):
        run_predict(adapter, "3.5", field="rating", sig=RatingSig)


def test_scale_construction_errors():
    with pytest.raises(TypeError):
        modaic.Scale[1]
    with pytest.raises(TypeError):
        modaic.Scale[1, 2, 3]
    with pytest.raises(TypeError):
        modaic.Scale["a", "b"]
    with pytest.raises(TypeError):
        modaic.Scale[True, 5]
    with pytest.raises(ValueError):
        modaic.Scale[5, 1]


def test_scale_annotation_shape():
    from typing import Literal

    ann = modaic.Scale[1, 5]
    # DSPy reads these duck-typed attrs directly (same convention as modaic.Enum)
    assert ann.__origin__ is Literal
    assert ann.__args__ == (1, 2, 3, 4, 5)
