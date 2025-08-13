from typing import Union, Optional, TypeAlias, Literal
from typing import NamedTuple, Type
from types import NoneType
from colorama import Fore, Back, Style, init

init(autoreset=True)


ValueType: TypeAlias = Union[int, str, float, bool, NoneType, list, "Value"]

mql_operator_to_python = {
    "$eq": "==",
    "$lt": "<",
    "$le": "<=",
    "$gt": ">",
    "$ge": ">=",
    "$ne": "!=",
}


def print_return(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if isinstance(op := args[1], str) and op[0] == "$":
            if kwargs.get("recursed", False):
                print(
                    f"{repr(args[0])} ({mql_operator_to_python[op]}) {repr(args[2])} ->:",
                    result,
                )
            else:
                print(
                    f"{repr(args[0])} {mql_operator_to_python[op]} {repr(args[2])} ->:",
                    result,
                )
        else:
            if func.__name__ == "__and__":
                print(f"{repr(args[0])} & {repr(args[1])} ->:", result)
            elif func.__name__ == "__rand__":
                print(f"{repr(args[1])} & {repr(args[0])} ->:", result)
        return result

    return wrapper


class QueryParam:
    """
    Modaic Query Language Property class.
    """

    type: Optional[str] = None

    def __init__(
        self, *args, query: Optional[dict] = None, complete: bool = False, **kwargs
    ):
        self.complete = complete
        self.query = query or {}

    def __repr__(self):
        if isinstance(self, Prop):
            return f"Prop('{self.name}')"
        elif isinstance(self, Value):
            return f"Value({self.value})"
        elif isinstance(self, AND):
            left_style = Fore.RED if not self.left.complete else Fore.GREEN
            right_style = Fore.RED if not self.right.complete else Fore.GREEN
            return f"AND({left_style}{repr(self.left)}{Style.RESET_ALL}, {right_style}{repr(self.right)}{Style.RESET_ALL})"
        elif isinstance(self, OR):
            left_style = Fore.RED if not self.left.complete else Fore.GREEN
            right_style = Fore.RED if not self.right.complete else Fore.GREEN
            return f"OR({repr(self.left)}, {repr(self.right)})"
        else:
            return str(self.query)

    def __eq__(self, other: Union[ValueType, "QueryParam"]):
        return self.comparison("$eq", other)

    def __lt__(self, other: Union[ValueType, "QueryParam"]):
        return self.comparison("$lt", other)

    def __le__(self, other: Union[ValueType, "QueryParam"]):
        return self.comparison("$le", other)

    def __gt__(self, other: Union[ValueType, "QueryParam"]):
        return self.comparison("$gt", other)

    def __ge__(self, other: Union[ValueType, "QueryParam"]):
        return self.comparison("$ge", other)

    def __ne__(self, other: Union[ValueType, "QueryParam"]):
        return self.comparison("$ne", other)

    def __bool__(self):
        # return self
        raise ValueError("QueryParam cannot be used as a boolean")

    @print_return
    def __and__(self, other: Union["QueryParam", ValueType]):
        if not isinstance(other, QueryParam):
            other = Value(other)
        return AND(self, other)

    @print_return
    def __or__(self, other: Union["QueryParam", ValueType]):
        if not isinstance(other, QueryParam):
            other = Value(other)
        return OR(self, other)

    @print_return
    def __rand__(self, other: Union[int, str, list]):
        if not isinstance(other, QueryParam):
            other = Value(other)
        return AND(other, self)

    @print_return
    def __ror__(self, other: Union[int, str, list]):
        if not isinstance(other, QueryParam):
            other = Value(other)
        return OR(other, self)

    @print_return
    def comparison(
        self,
        op: str,
        other: Union[ValueType, "QueryParam"],
        recursed: bool = False,
    ):
        # Check that no completed boolean expressions are used in the comparison
        enforce_types(other, op, allowed_types[op])

        if not isinstance(other, QueryParam):
            other = Value(other)

        if isinstance(self, AND) and not self.complete:
            if not self.right.complete:
                print(
                    "fake and (self)",
                    f"{self.left} & ({self.right} {mql_operator_to_python[op]} {other})",
                )
                return self.left & self.right.comparison(op, other, recursed=True)
        elif isinstance(self, OR) and not self.complete:
            if not self.right.complete:
                print("fake or (self)", self, "with", other)
                return self.left | self.right.comparison(op, other, recursed=True)
        elif not isinstance(self, Prop):
            raise ValueError(f"Right hand side must be a property, got {type(self)}")

        if isinstance(other, AND) and not other.complete:
            if not other.left.complete:
                print(
                    "fake and (other)",
                    f"({self} {mql_operator_to_python[op]} {other.left}) & {other.right}",
                )
                return self.comparison(op, other.left, recursed=True) & other.right
        elif isinstance(other, OR) and not other.complete:
            if not other.left.complete:
                print("fake or (other)", other)
                return self.comparison(op, other.left, recursed=True) | other.right
        elif isinstance(other, Value):
            return QueryParam(query={self.name: {op: other.value}}, complete=True)
        elif isinstance(self, Prop) and isinstance(other, Prop):
            return QueryParam(query={op: {self.name: other.name}}, complete=True)

        raise ValueError(f"Invalid comparison type: {type(other)}")

    def rcomparison(
        self,
        op: str,
        other: Union[ValueType, "QueryParam"],
    ):
        enforce_types(other, op, allowed_types[op])
        if not isinstance(other, QueryParam):
            other = Value(other)
        if isinstance(other, AND) and not other.complete:
            if not other.right.complete:
                return other.left & self.rcomparison(op, other.right)
        elif isinstance(other, OR) and not other.complete:
            if not other.left.complete:
                return other.right | self.rcomparison(op, other.left)
        else:
            return QueryParam(query={self.name: {op: other}}, complete=True)


def enforce_types(
    other: ValueType,
    op: str,
    enforced_types: list[Type] = None,
):
    if enforced_types is None:
        return
    other_type = get_type(other)
    if other_type not in enforced_types:
        raise ValueError(
            f"Value must be one of {enforced_types}, got {other_type} for {op}"
        )


def get_type(value: ValueType, value_side="right"):
    if isinstance(value, Value):
        return type(value.value)
    elif isinstance(value, QueryParam) and value.complete:
        return bool
    elif isinstance(value, (AND, OR)) and not value.complete:
        # if value is on the right side of the operator, return the type of the left side of the uncompleted expression
        if value_side == "right":
            return get_type(value.left, "left")
        elif value_side == "left":
            return get_type(value.right, "right")
    else:
        return type(value)


class Prop(QueryParam):
    """
    Modaic Query Language Property class.
    """

    type = "prop"

    def __init__(
        self,
        name: str,
    ):
        super().__init__()
        self.name = name


class Value(QueryParam):
    """
    Modaic Query Language Value class.
    """

    type = "value"

    def __init__(
        self,
        value: int | str | list,
    ):
        super().__init__()
        self.value = value


class AND(QueryParam):
    """
    Modaic Query Language AND class.
    """

    type = "and"

    def __init__(self, left: "QueryParam", right: "QueryParam"):
        super().__init__()
        self.left = left
        self.right = right
        # if complete, we can set the query other wise parent - QueryParam will make it {}
        if self.left.complete and self.right.complete:
            self.query = {"$and": [self.left.query, self.right.query]}
            self.complete = True


class OR(QueryParam):
    """
    Modaic Query Language OR class.
    """

    type = "or"

    def __init__(self, left: "QueryParam", right: "QueryParam", complete: bool = False):
        super().__init__()
        self.left = left
        self.right = right
        # if complete, we can set the query other wise parent - QueryParam will make it {}
        if self.left.complete and self.right.complete:
            self.query = {"$or": [self.left.query, self.right.query]}
            self.complete = True


allowed_types = {
    "$eq": [int, str, list, dict, bool, NoneType, Prop],
    "$lt": [int, float, Prop],
    "$le": [int, float, Prop],
    "$gt": [int, float, Prop],
    "$ge": [int, float, Prop],
    "$ne": [int, str, list, dict, bool, NoneType, Prop],
}
