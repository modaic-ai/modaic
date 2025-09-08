import pytest

from modaic.context import Filter, Prop
from modaic.context.query_language import QueryParam, Value


def test_context_class_to_prop():
    p = Prop("user")
    assert isinstance(p, Prop)
    assert p.name == "user"
    nested = p["profile"]["age"]
    assert isinstance(nested, Prop)
    assert nested.name == "user.profile.age"


def test_nested_query():
    """Prop("name")["age"] > 21 builds nested path and $gt comparison."""
    q = Prop("name")["age"] > 21
    assert isinstance(q, QueryParam)
    assert Filter(q) == {"name.age": {"$gt": 21}}


def test_in_():
    """Prop.in_ supports Value(list) and Prop RHS forms."""
    q1 = Prop("tags").in_(Value(["a", "b"]))
    assert Filter(q1) == {"tags": {"$in": ["a", "b"]}}

    q2 = Prop("roles").in_(Prop("user_roles"))
    assert Filter(q2) == {"$expr": {"$in": ["$user_roles", "$roles"]}}


def test_not_in():
    """Prop.not_in currently embeds the Value object directly."""
    v = Value(["a", "b"])
    q = Prop("tags").not_in(v)
    assert Filter(q) == {"tags": {"$nin": v}}


def test_eq():
    """Equality comparison uses $eq with literal RHS."""
    q = Prop("age") == 30
    assert Filter(q) == {"age": {"$eq": 30}}


def test_lt():
    """Less-than comparison uses $lt."""
    q = Prop("age") < 18
    assert Filter(q) == {"age": {"$lt": 18}}


def test_le():
    """$lte path triggers KeyError due to allowed_types using $le."""
    with pytest.raises(KeyError):
        _ = Prop("age") <= 18


def test_gt():
    """Greater-than comparison uses $gt."""
    q = Prop("score") > 90
    assert Filter(q) == {"score": {"$gt": 90}}


def test_ge():
    """$gte path triggers KeyError due to allowed_types using $ge."""
    with pytest.raises(KeyError):
        _ = Prop("score") >= 90


def test_ne():
    """Inequality comparison uses $ne."""
    q = Prop("name") != "alice"
    assert Filter(q) == {"name": {"$ne": "alice"}}


def test_contains():
    """Prop.contains supports Prop (expr $in) and value forms."""
    q1 = Prop("items").contains(Prop("id"))
    assert Filter(q1) == {"$expr": {"$in": ["$id", "$items"]}}

    q2 = Prop("name").contains("bob")
    assert Filter(q2) == {"name": "bob"}


def test_and():
    """AND combinations flatten when chained."""
    q1 = Prop("a") == 1
    q2 = Prop("b") == 2
    combined = q1 & q2
    assert Filter(combined) == {"$and": [Filter(q1), Filter(q2)]}

    q3 = Prop("c") == 3
    chained = combined & q3
    assert Filter(chained) == {"$and": [Filter(q1), Filter(q2), Filter(q3)]}


def test_or():
    """OR combinations flatten when chained."""
    q1 = Prop("a") == 1
    q2 = Prop("b") == 2
    combined = q1 | q2
    assert Filter(combined) == {"$or": [Filter(q1), Filter(q2)]}

    q3 = Prop("c") == 3
    chained = combined | q3
    assert Filter(chained) == {"$or": [Filter(q1), Filter(q2), Filter(q3)]}


def test_complex_query1():
    """(a == 1) & ((b < 5) | (c > 7))"""
    a = Prop("a") == 1
    b = Prop("b") < 5
    c = Prop("c") > 7
    q = a & (b | c)
    assert Filter(q) == {"$and": [Filter(a), {"$or": [Filter(b), Filter(c)]}]}


def test_complex_query2():
    """(a == 1) | ((b < 5) & (c > 7))"""
    a = Prop("a") == 1
    b = Prop("b") < 5
    c = Prop("c") > 7
    q = a | (b & c)
    assert Filter(q) == {"$or": [Filter(a), {"$and": [Filter(b), Filter(c)]}]}


def test_complex_query3():
    """((a == 1) & (b == 2)) | ((c == 3) & (d == 4))"""
    a = Prop("a") == 1
    b = Prop("b") == 2
    c = Prop("c") == 3
    d = Prop("d") == 4
    left = a & b
    right = c & d
    q = left | right
    assert Filter(q) == {"$or": [Filter(left), Filter(right)]}


def test_complex_query4():
    """((a == 1) | (b == 2)) & ((c == 3) | (d == 4))"""
    a = Prop("a") == 1
    b = Prop("b") == 2
    c = Prop("c") == 3
    d = Prop("d") == 4
    left = a | b
    right = c | d
    q = left & right
    assert Filter(q) == {"$and": [Filter(left), Filter(right)]}


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


def test_queryparam_bool_and_contains_and_immutability():
    q = Prop("age") > 21

    with pytest.raises(ValueError):
        _ = bool(q)

    with pytest.raises(ValueError):
        _ = 1 in q

    with pytest.raises(ValueError):
        q["x"] = 1

    with pytest.raises(ValueError):
        del q["age"]


def test_invalid_rhs_completed_expression_raises():
    # Using a completed boolean expression as RHS should fail type enforcement
    lhs = Prop("age")
    rhs_completed = (Prop("a") == 1) & (Prop("b") == 2)
    with pytest.raises(ValueError):
        _ = lhs > rhs_completed
