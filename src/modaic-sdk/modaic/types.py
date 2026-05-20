# ruff: noqa: ANN001
from typing import Literal

from pydantic_core import core_schema


class _EnumAnnotation:
    """
    Returned by Enum.__class_getitem__. Acts as a type annotation that DSPy and pydantic
    both understand — DSPy sees it as a Literal (correct prompt generation), pydantic uses
    __get_pydantic_core_schema__ to validate with normalization applied first.
    """

    def __init__(self, values: tuple):
        self.__origin__ = Literal
        self.__args__ = values

    def __get_pydantic_core_schema__(self, source, handler):
        allowed = self.__args__

        def validate(v):
            # json_repair parses "[YES]" as ["YES"] before we see it — unwrap single-element lists
            if isinstance(v, list) and len(v) == 1:
                v = str(v[0])

            if not isinstance(v, str):
                return v

            v = v.strip()

            # Strip one layer of wrapping parens — brackets are handled via the list unwrap above
            if v.startswith("(") and v.endswith(")"):
                v = v[1:-1].strip()

            # Strip leading/trailing dots
            v = v.strip(".").strip()

            # Collapse repeated single character: "AAAA" -> "A", "aaaa" -> "a"
            if len(v) > 1 and len(set(v.upper())) == 1:
                v = v[0]

            if v in allowed:
                return v

            # Case-insensitive match — return the original-cased allowed value
            v_lower = v.lower()
            for a in allowed:
                if a.lower() == v_lower:
                    return a

            raise ValueError(f"{v!r} is not one of {allowed!r}")

        return core_schema.no_info_plain_validator_function(validate)

    def __get_pydantic_json_schema__(self, schema, handler):
        # Delegate to the underlying Literal so model_json_schema() can serialize
        # this annotation (e.g. when a judge gets pushed to modaic Hub).
        return handler(core_schema.literal_schema(list(self.__args__)))

    def __repr__(self):
        args = ", ".join(repr(v) for v in self.__args__)
        return f"modaic.Enum[{args}]"


class Enum:
    """A Literal-like type annotation that normalizes noisy LLM outputs before validation.

    Handles bracket wrapping ("(A)" or "[A]"), repeated single characters ("AAAA"),
    case insensitivity, whitespace, and trailing dots.

    Usage::

        class MySignature(dspy.Signature):
            decision: modaic.Enum["YES", "NO", "MAYBE"] = dspy.OutputField()
    """

    def __class_getitem__(cls, values):
        if not isinstance(values, tuple):
            values = (values,)
        return _EnumAnnotation(values)


class _ScaleAnnotation:
    """
    Returned by Scale.__class_getitem__. Acts as a type annotation that DSPy and pydantic
    both understand — DSPy sees it as a Literal of ints (so the model emits unquoted
    integers chosen from the range), pydantic uses __get_pydantic_core_schema__ to
    coerce noisy outputs to int and range-check.
    """

    def __init__(self, lo: int, hi: int):
        self.lo = lo
        self.hi = hi
        self.__origin__ = Literal
        self.__args__ = tuple(range(lo, hi + 1))

    def __get_pydantic_core_schema__(self, source, handler):
        lo, hi = self.lo, self.hi

        def validate(v):
            # json_repair parses "[3]" as [3] before we see it — unwrap single-element lists
            if isinstance(v, list) and len(v) == 1:
                v = v[0]

            # bool is a subclass of int in Python — reject so True/False don't silently become 1/0
            if isinstance(v, bool):
                raise ValueError(f"{v!r} is not a valid integer")

            if isinstance(v, int):
                n = v
            elif isinstance(v, float):
                if not v.is_integer():
                    raise ValueError(f"{v!r} is not a valid integer")
                n = int(v)
            elif isinstance(v, str):
                s = v.strip()

                # Strip one layer of wrapping parens or brackets
                if (s.startswith("(") and s.endswith(")")) or (s.startswith("[") and s.endswith("]")):
                    s = s[1:-1].strip()

                # Strip leading/trailing dots
                s = s.strip(".").strip()

                try:
                    n = int(s)
                except ValueError:
                    try:
                        f = float(s)
                    except ValueError:
                        raise ValueError(f"{v!r} is not a valid integer") from None
                    if not f.is_integer():
                        raise ValueError(f"{v!r} is not a valid integer") from None
                    n = int(f)
            else:
                raise ValueError(f"{v!r} is not a valid integer")

            if not (lo <= n <= hi):
                raise ValueError(f"{n} is not in range [{lo}, {hi}]")

            return n

        return core_schema.no_info_plain_validator_function(validate)

    def __get_pydantic_json_schema__(self, schema, handler):
        # Delegate to the underlying Literal of ints so model_json_schema() can
        # serialize this annotation (e.g. when a judge gets pushed to modaic Hub).
        return handler(core_schema.literal_schema(list(self.__args__)))

    def __repr__(self):
        return f"modaic.Scale[{self.lo}, {self.hi}]"


class Scale:
    """A Literal-like type annotation for an integer scale ``[lo, hi]`` (inclusive).

    Presents to DSPy as ``Literal[lo, lo+1, ..., hi]`` of ints, so the model emits
    unquoted integers. Validation is loose: string outputs like ``"3"`` or ``"3."``,
    bracketed forms like ``"(3)"``, and integral floats like ``3.0`` are coerced
    to int before the range check.

    Usage::

        class MySignature(dspy.Signature):
            rating: modaic.Scale[1, 5] = dspy.OutputField()
    """

    def __class_getitem__(cls, values):
        if not isinstance(values, tuple) or len(values) != 2:
            raise TypeError("Scale requires exactly 2 values: Scale[lo, hi]")
        lo, hi = values
        if isinstance(lo, bool) or isinstance(hi, bool) or not isinstance(lo, int) or not isinstance(hi, int):
            raise TypeError("Scale values must be integers")
        if lo > hi:
            raise ValueError(f"Scale lo ({lo}) must be <= hi ({hi})")
        return _ScaleAnnotation(lo, hi)
