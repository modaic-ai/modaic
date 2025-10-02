from types import NoneType
from typing import Optional, Type, TypeAlias, Union

from langchain_core.structured_query import Comparator, Comparison, Operation, Operator, Visitor

ValueType: TypeAlias = Union[int, str, float, bool, NoneType, list, "Value"]

value_types = (int, str, float, bool, NoneType, list)

comparator_to_python = {
    Comparator.EQ: "==",
    Comparator.LT: "<",
    Comparator.LTE: "<=",
    Comparator.GT: ">",
    Comparator.GTE: ">=",
    Comparator.NE: "!=",
    Comparator.CONTAIN: "contains",
    Comparator.IN: "in_",
    Comparator.NIN: "not_in",
}

allowed_types = {
    Comparator.EQ: [int, str, list, dict, bool, NoneType],
    Comparator.LT: [int, float],
    Comparator.LTE: [int, float],
    Comparator.GT: [int, float],
    Comparator.GTE: [int, float],
    Comparator.NE: [int, str, list, dict, bool, NoneType],
    Comparator.CONTAIN: [int, str, bool, NoneType],
    Comparator.IN: [list, dict],
    Comparator.NIN: [list, dict],
}


class Condition:
    """
    Modaic Query Language Property class.
    """

    is_operation: bool = False
    condition: Comparison | Operation = None

    def __init__(self, condition: Comparison | Operation):
        if isinstance(condition, Comparison):
            self.is_operation = False
        else:
            self.is_operation = True
        self.condition = condition

    def __repr__(self):
        return f"Condition({repr(self.condition)})"

    def __str__(self):
        return (
            f"Prop({self.condition.attribute}) {comparator_to_python[self.condition.comparator]} {self.condition.value}"
        )

    def __contains__(self, other: str):
        raise ValueError("Modaic Filters do not support `in` use Prop.in_()/Prop.not_in() instead")

    def __bool__(self):
        raise ValueError(
            "Attempted to evaluate Modaic condition as boolean. Please make sure you wrap ALL expresions with ()"
        )

    def __and__(self, other: Union["Condition", ValueType]):
        if not isinstance(other, Condition):
            other = Value(other)
        return AND(self, other)

    # @_print_return
    def __or__(self, other: Union["Condition", ValueType]):
        if not isinstance(other, Condition):
            other = Value(other)
        return OR(self, other)

    # @_print_return
    def __rand__(self, other: Union[int, str, list]):
        if not isinstance(other, Condition):
            other = Value(other)
        return AND(other, self)

    # @_print_return
    def __ror__(self, other: Union[int, str, list]):
        if not isinstance(other, Condition):
            other = Value(other)
        return OR(other, self)

    def __invert__(self):
        # TODO: implement , use nor
        raise NotImplementedError("Modaic queires do not support ! yet")


def _enforce_types(
    other: ValueType,
    op: str,
    allowed_types: list[Type],
):
    other_type = _get_type(other)
    if other_type not in allowed_types:
        raise ValueError(f"Value must be one of {allowed_types}, got {other_type} for {op}")


def _get_type(value: ValueType):
    if isinstance(value, Value):
        return type(value.value)
    elif isinstance(value, value_types):
        return type(value)
    elif isinstance(value, Condition):
        return bool
    elif isinstance(value, Prop):
        raise NotImplementedError("Modaic queries do not support comparisions between properties yet.")
    else:
        raise ValueError(f"Unssuported value type in Modaic condition, got {type(value)}")


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
        raise NotImplementedError("Modaic queries do not support nested properties yet")

    def in_(self, other: Union["Value", list]) -> "Condition":
        value = Value(other)
        return self.comparison("in", Comparator.IN, value)

    def not_in(self, other: Union["Value", list]) -> "Condition":
        value = Value(other)
        return self.comparison("not_in", Comparator.NIN, value)

    def __eq__(self, other: Optional[Union[ValueType]]) -> "Condition":
        value = Value(other)
        return self.comparison("==", Comparator.EQ, value)

    def __lt__(self, other: Union[ValueType]) -> "Condition":
        value = Value(other)
        return self.comparison("<", Comparator.LT, value)

    def __le__(self, other: Union[ValueType]) -> "Condition":
        value = Value(other)
        return self.comparison("<=", Comparator.LTE, value)

    def __gt__(self, other: Union[ValueType]) -> "Condition":
        value = Value(other)
        return self.comparison(">", Comparator.GT, value)

    def __ge__(self, other: Union[ValueType]) -> "Condition":
        value = Value(other)
        return self.comparison(">=", Comparator.GTE, value)

    def __ne__(self, other: Optional[Union[ValueType]]) -> "Condition":
        value = Value(other)
        return self.comparison("!=", Comparator.NE, value)

    def contains(self, other: Union[ValueType]) -> "Condition":
        value = Value(other)
        return self.comparison("contains", Comparator.CONTAIN, value)

    def exists(self) -> "Condition":
        # TODO: implement
        raise NotImplementedError("Prop does not support exists")

    def not_exists(self) -> "Condition":
        # TODO: implement
        raise NotImplementedError("Prop does not support not_exists")

    def comparison(self, op_expr: str, comparator: Comparator, other: "Value") -> "Condition":
        _enforce_types(other, op_expr, allowed_types[comparator])
        comparison = Comparison(
            comparator=comparator,
            attribute=self.name,
            value=other.value,
        )
        return Condition(comparison)

    def __len__(self):
        # TODO: implement
        raise NotImplementedError("Prop does not support __len__")

    def all(self, other):  # noqa: ANN001
        # TODO: implement
        raise NotImplementedError("Prop does not support all")

    def any(self, other):  # noqa: ANN001
        # TODO: implement
        raise NotImplementedError("Prop does not support any")


class Value:
    """
    Modaic Query Language Value class.
    """

    def __init__(self, value: Union[int, str, list, dict, bool, "Value", None]):
        super().__init__()
        if not isinstance(value, ValueType):
            raise ValueError(
                f"Value must be one of {value_types}, got {type(value)}. Please wrap your expressions with ()"
            )
        if isinstance(value, Value):
            value = value.value
        self.value = value


class AND(Condition):
    """
    Modaic Query Language AND class.
    """

    is_operation: bool = True
    condition: Operation

    def __init__(self, left: "Condition", right: "Condition"):
        self.left = left
        self.right = right
        if isinstance(self.left, AND) and isinstance(self.right, AND):
            arguments = self.left.condition.arguments + self.right.condition.arguments
        elif and_other := _get_and_other(self.left, self.right):
            arguments = and_other[0].condition.arguments + [and_other[1].condition]
        else:
            arguments = [self.left.condition, self.right.condition]
        super().__init__(Operation(operator=Operator.AND, arguments=arguments))

    def __repr__(self):
        return f"AND({self.left}, {self.right})"

    def __str__(self):
        return f"({str(self.left)}) & ({str(self.right)})"


class OR(Condition):
    """
    Modaic Query Language OR class.
    """

    is_operation: bool = True
    condition: Operation

    def __init__(self, left: "Condition", right: "Condition", complete: bool = False):
        self.left = left
        self.right = right
        if isinstance(self.left, OR) and isinstance(self.right, OR):
            arguments = self.left.condition.arguments + self.right.condition.arguments
        elif or_other := _get_or_other(self.left, self.right):
            arguments = or_other[0].condition.arguments + [or_other[1].condition]
        else:
            arguments = [self.left.condition, self.right.condition]
        super().__init__(Operation(operator=Operator.OR, arguments=arguments))

    def __repr__(self):
        return f"OR({self.left}, {self.right})"

    def __str__(self):
        return f"({str(self.left)}) | ({str(self.right)})"


def _get_and_or(left: "Condition", right: "Condition"):
    if isinstance(left, AND) and isinstance(right, OR):
        return left, right
    elif isinstance(right, AND) and isinstance(left, OR):
        return right, left
    else:
        return None


def _get_and_other(left: "Condition", right: "Condition"):
    if isinstance(left, AND) and type(right) is Condition:
        return left, right
    elif isinstance(right, AND) and type(left) is Condition:
        return right, left
    else:
        return None


def _get_or_other(left: "Condition", right: "Condition"):
    if isinstance(left, OR) and right is Condition:
        return left, right
    elif isinstance(right, OR) and left is Condition:
        return right, left
    else:
        return None


def parse_modaic_filter(translator: Visitor, condition: Condition) -> dict:  # noqa: N802
    if condition.is_operation:
        return translator.visit_operation(condition.condition)
    else:
        return translator.visit_comparison(condition.condition)
