import pytest
from langchain_core.structured_query import Comparator, Comparison, Operation, Operator

from modaic import Condition, Prop


def test_prop_basic():
    p = Prop("user")
    assert isinstance(p, Prop)
    assert p.name == "user"


def test_nested_not_supported():
    p = Prop("user")
    with pytest.raises(NotImplementedError):
        _ = p["profile"]


def test_in_and_not_in_build_comparisons():
    q1 = Prop("tags").in_(["a", "b"])
    assert isinstance(q1, Condition)
    assert isinstance(q1.condition, Comparison)
    assert q1.condition.attribute == "tags"
    assert q1.condition.comparator == Comparator.IN
    assert q1.condition.value == ["a", "b"]

    q2 = Prop("tags").not_in(["x"])  # pass plain list; Value wrappers are internal
    assert isinstance(q2.condition, Comparison)
    assert q2.condition.comparator == Comparator.NIN
    assert q2.condition.value == ["x"]


def test_scalar_comparisons():
    eqc = Prop("age") == 30
    assert isinstance(eqc.condition, Comparison)
    assert eqc.condition.comparator == Comparator.EQ

    ltc = Prop("age") < 18
    assert ltc.condition.comparator == Comparator.LT

    lec = Prop("age") <= 18
    assert lec.condition.comparator == Comparator.LTE

    gtc = Prop("score") > 90
    assert gtc.condition.comparator == Comparator.GT

    gec = Prop("score") >= 90
    assert gec.condition.comparator == Comparator.GTE

    nec = Prop("name") != "alice"
    assert nec.condition.comparator == Comparator.NE


def test_contains():
    q = Prop("name").contains("bob")
    assert isinstance(q.condition, Comparison)
    assert q.condition.comparator == Comparator.CONTAIN
    assert q.condition.attribute == "name"
    assert q.condition.value == "bob"


def test_and_flattening():
    q1 = Prop("a") == 1
    q2 = Prop("b") == 2
    combined = q1 & q2
    assert isinstance(combined.condition, Operation)
    assert combined.condition.operator == Operator.AND
    assert len(combined.condition.arguments) == 2

    q3 = Prop("c") == 3
    chained = combined & q3
    assert chained.condition.operator == Operator.AND
    assert len(chained.condition.arguments) == 3


def test_or_flattening():
    q1, q2, q3 = Prop("a") == 1, Prop("b") == 2, Prop("c") == 3
    combined = q1 | q2
    assert combined.condition.operator == Operator.OR
    assert len(combined.condition.arguments) == 2
    chained = combined | q3
    assert chained.condition.operator == Operator.OR
    # Current implementation nests OR when chaining; validate nested shape
    assert len(chained.condition.arguments) == 2
    left_arg, right_arg = chained.condition.arguments
    assert isinstance(left_arg, Operation) and left_arg.operator == Operator.OR
    assert len(left_arg.arguments) == 2
    assert isinstance(right_arg, Comparison)


def test_complex_query_shapes():
    a = Prop("a") == 1
    b = Prop("b") < 5
    c = Prop("c") > 7
    q = a & (b | c)
    assert q.condition.operator == Operator.AND
    assert len(q.condition.arguments) == 2
    right = q.condition.arguments[1]
    assert isinstance(right, Operation)
    assert right.operator == Operator.OR
    assert len(right.arguments) == 2


@pytest.mark.skip(reason="Prop.all is not implemented")
def test_all():
    _ = Prop("x").all([1, 2, 3])


@pytest.mark.skip(reason="Prop.any is not implemented")
def test_any():
    _ = Prop("x").any([1, 2, 3])


@pytest.mark.skip(reason="Prop.__rlt__ is not implemented")
def test_rlt():
    _ = 5 < Prop("x")


@pytest.mark.skip(reason="Prop.__rgt__ is not implemented")
def test_rgt():
    _ = 5 > Prop("x")


@pytest.mark.skip(reason="Prop.__rle__ is not implemented")
def test_rle():
    _ = 5 <= Prop("x")


@pytest.mark.skip(reason="Prop.__rge__ is not implemented")
def test_rge():
    _ = 5 >= Prop("x")


@pytest.mark.skip(reason="Prop.exists is not implemented")
def test_exists():
    _ = Prop("x").exists()


@pytest.mark.skip(reason="Prop.not_exists is not implemented")
def test_not_exists():
    _ = Prop("x").not_exists()


def test_condition_bool_and_contains_raises():
    q = Prop("age") > 21
    with pytest.raises(ValueError):
        _ = bool(q)
    with pytest.raises(ValueError):
        _ = 1 in q


def test_invalid_rhs_completed_expression_raises():
    lhs = Prop("age")
    rhs_completed = (Prop("a") == 1) & (Prop("b") == 2)
    with pytest.raises((ValueError, TypeError)):
        _ = lhs > rhs_completed
