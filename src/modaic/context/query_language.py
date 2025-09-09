from types import NoneType
from typing import Any, Literal, Optional, Type, TypeAlias, Union

ValueType: TypeAlias = Union[int, str, float, bool, NoneType, list, "Value"]

value_types = (int, str, float, bool, NoneType, list)

mql_operator_to_python = {
    "$eq": "==",
    "$lt": "<",
    "$le": "<=",
    "$gt": ">",
    "$ge": ">=",
    "$ne": "!=",
}


def _print_return(func):  # noqa: ANN001
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if isinstance(op := args[1], str) and op[0] == "$":
            if kwargs.get("recursed", False):
                print(  # noqa: T201
                    f"{repr(args[0])} ({mql_operator_to_python[op]}) {repr(args[2])} ->:",
                    result,
                )
            else:
                print(  # noqa: T201
                    f"{repr(args[0])} {mql_operator_to_python[op]} {repr(args[2])} ->:",
                    result,
                )
        else:
            if func.__name__ == "__and__":
                print(f"{repr(args[0])} & {repr(args[1])} ->:", result)  # noqa: T201
            elif func.__name__ == "__rand__":
                print(f"{repr(args[1])} & {repr(args[0])} ->:", result)  # noqa: T201
        return result

    return wrapper


class QueryParam:
    """
    Modaic Query Language Property class.
    """

    type: Optional[str] = None

    def __init__(self, *args, query: Optional[dict] = None, **kwargs):
        self.query = query or {}

    def __repr__(self):
        return f"QueryParam({self.query})"

    def __contains__(self, other: str):
        raise ValueError("Modaic Queries do not support `in` use Prop.in_()/Prop.not_in() instead")

    def __bool__(self):
        raise ValueError(
            "Attempted to evaluate Modaic Query as boolean. Please make sure you wrap ALL expresions with ()"
        )

    def __and__(self, other: Union["QueryParam", ValueType]):
        if not isinstance(other, QueryParam):
            other = Value(other)
        return AND(self, other)

    # @_print_return
    def __or__(self, other: Union["QueryParam", ValueType]):
        if not isinstance(other, QueryParam):
            other = Value(other)
        return OR(self, other)

    # @_print_return
    def __rand__(self, other: Union[int, str, list]):
        if not isinstance(other, QueryParam):
            other = Value(other)
        return AND(other, self)

    # @_print_return
    def __ror__(self, other: Union[int, str, list]):
        if not isinstance(other, QueryParam):
            other = Value(other)
        return OR(other, self)

    def __invert__(self):
        # TODO: implement , use nor
        pass

    def __getitem__(self, key: str):
        return self.query[key]

    def __iter__(self):
        return iter(self.query)

    def __len__(self):
        return len(self.query)

    def __setitem__(self, key: str, value: Any):
        raise ValueError("QueryParam is immutable")

    def __delitem__(self, key: str):
        raise ValueError("QueryParam is immutable")


def _enforce_types(
    other: ValueType,
    op: str,
    enforced_types: list[Type] = None,
):
    if enforced_types is None:
        return
    other_type = _get_type(other)
    if other_type not in enforced_types:
        raise ValueError(f"Value must be one of {enforced_types}, got {other_type} for {op}")


def _get_type(value: ValueType):
    if isinstance(value, Value):
        return type(value.value)
    elif isinstance(value, value_types):
        return type(value)
    elif isinstance(value, Prop):
        return Prop
    elif isinstance(value, QueryParam):
        return bool


class Prop:
    """
    Modaic Query Language Property class.
    """

    def __init__(
        self,
        name: str,
    ):
        super().__init__()
        self.name = name

    def __getitem__(self, key: str):
        return Prop(f"{self.name}.{key}")

    def in_(self, other: Union["Prop", "Value"]) -> "QueryParam":
        if isinstance(other, Value):
            return QueryParam(query={self.name: {"$in": other.value}})
        elif isinstance(other, Prop):
            return QueryParam(query={"$expr": {"$in": [f"${other.name}", f"${self.name}"]}})
        else:
            raise ValueError(
                f"Right hand side of in must be a property or value, got {type(other)}. Please wrap your expressions with ()"
            )

    def not_in(self, other: Union["Prop", "Value"]) -> "QueryParam":
        return QueryParam(query={self.name: {"$nin": other}})

    def __eq__(self, other: Optional[Union[ValueType, "Prop"]]):
        return self.comparison("$eq", other)

    def __lt__(self, other: Union[ValueType, "Prop"]):
        return self.comparison("$lt", other)

    def __le__(self, other: Union[ValueType, "Prop"]):
        return self.comparison("$lte", other)

    def __gt__(self, other: Union[ValueType, "Prop"]):
        return self.comparison("$gt", other)

    def __ge__(self, other: Union[ValueType, "Prop"]):
        return self.comparison("$gte", other)

    def __ne__(self, other: Optional[Union[ValueType, "Prop"]]):
        return self.comparison("$ne", other)

    def contains(self, other: Union[ValueType, "Prop"]) -> "QueryParam":
        if isinstance(other, value_types):
            other = Value(other)
        if isinstance(other, Prop):
            return other.in_(self)
        else:
            return QueryParam(query={self.name: other.value})

    def __len__(self):
        raise NotImplementedError("Prop does not support __len__")

    def all(self, other):  # noqa: ANN001
        # TODO: implement
        raise NotImplementedError("Prop does not support all")

    def any(self, other):  # noqa: ANN001
        # TODO: implement
        raise NotImplementedError("Prop does not support any")

    def __rlt__(self, other: ValueType):
        # TODO: implement
        raise NotImplementedError("Prop does not support __rlt__")

    def __rgt__(self, other: ValueType):
        # TODO: implement
        raise NotImplementedError("Prop does not support __rgt__")

    def __rle__(self, other: ValueType):
        # TODO: implement
        raise NotImplementedError("Prop does not support __rle__")

    def __rge__(self, other: ValueType):
        # TODO: implement
        raise NotImplementedError("Prop does not support __rge__")

    def exists(self):
        # TODO: implement
        raise NotImplementedError("Prop does not support exists")

    def not_exists(self):
        # TODO: implement
        raise NotImplementedError("Prop does not support not_exists")

    # @_print_return
    def comparison(
        self,
        op: str,
        other: Union[ValueType, "Prop"],
    ) -> "QueryParam":
        # Check that no completed boolean expressions are used in the comparison
        _enforce_types(other, op, allowed_types[op])

        if isinstance(other, value_types):
            other = Value(other)

        assert isinstance(self, Prop), (
            f"Left hand side of {mql_operator_to_python[op]} must be a property, got {type(self)}. Please wrap your expressions with ()"
        )

        if isinstance(other, Value):
            return QueryParam(query={self.name: {op: other.value}})
        elif isinstance(other, Prop):
            return QueryParam(query={op: [f"${self.name}", f"${other.name}"]})
        else:
            raise ValueError(
                f"Right hand side of {mql_operator_to_python[op]} must be a property or value, got {type(other)}. Please wrap your expressions with ()"
            )


class Value:
    """
    Modaic Query Language Value class.
    """

    def __init__(
        self,
        value: int | str | list | dict | bool | None,
    ):
        super().__init__()
        self.value = value


class AND(QueryParam):
    """
    Modaic Query Language AND class.
    """

    def __init__(self, left: "QueryParam", right: "QueryParam"):
        super().__init__()
        self.left = left
        self.right = right
        if isinstance(self.left, AND) and isinstance(self.right, AND):
            self.query = {"$and": self.left.query["$and"] + self.right.query["$and"]}
        elif and_other := _get_and_other(self.left, self.right):
            self.query = {"$and": and_other[0].query["$and"] + [and_other[1].query]}
        else:
            self.query = {"$and": [self.left.query, self.right.query]}

    def __repr__(self):
        return f"AND({self.left}, {self.right})"


class OR(QueryParam):
    """
    Modaic Query Language OR class.
    """

    def __init__(self, left: "QueryParam", right: "QueryParam", complete: bool = False):
        super().__init__()
        self.left = left
        self.right = right
        if isinstance(self.left, OR) and isinstance(self.right, OR):
            self.query = {"$or": self.left.query["$or"] + self.right.query["$or"]}
        elif or_other := _get_or_other(self.left, self.right):
            self.query = {"$or": or_other[0].query["$or"] + [or_other[1].query]}
        else:
            self.query = {"$or": [self.left.query, self.right.query]}

    def __repr__(self):
        return f"OR({self.left}, {self.right})"


def _get_and_or(left: "QueryParam", right: "QueryParam"):
    if isinstance(left, AND) and isinstance(right, OR):
        return left, right
    elif isinstance(right, AND) and isinstance(left, OR):
        return right, left
    else:
        return None


def _get_and_other(left: "QueryParam", right: "QueryParam"):
    if isinstance(left, AND) and type(right) is QueryParam:
        return left, right
    elif isinstance(right, AND) and type(left) is QueryParam:
        return right, left
    else:
        return None


def _get_or_other(left: "QueryParam", right: "QueryParam"):
    if isinstance(left, OR) and right is QueryParam:
        return left, right
    elif isinstance(right, OR) and left is QueryParam:
        return right, left
    else:
        return None


allowed_types = {
    "$eq": [int, str, list, dict, bool, NoneType, Prop],
    "$lt": [int, float, Prop],
    "$le": [int, float, Prop],
    "$gt": [int, float, Prop],
    "$ge": [int, float, Prop],
    "$ne": [int, str, list, dict, bool, NoneType, Prop],
}


def _build_in_check(
    left: "Prop",
    right: Union["Prop", "Value"],
    op: Literal["$in", "$nin"],
):
    if isinstance(left, Prop) and isinstance(right, Prop):
        return QueryParam(query={"$expr": {op: [f"${right.name}", f"${left.name}"]}})
    elif isinstance(right, Prop) and isinstance(left, Value):
        return QueryParam(query={right.name: {op: left.value}})
    else:
        raise ValueError(
            f"Right hand side of {op} must be a property or value, got {type(right)}. Please wrap your expressions with ()"
        )


def Filter(query: QueryParam) -> dict:  # noqa: N802
    return query.query
