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
