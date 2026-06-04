# ruff: noqa: ANN001
import functools
from typing import Literal

from pydantic_core import core_schema


class _LiteralLike(type):
    """Metaclass for the dynamically-built Scale/Enum annotation classes.

    The annotations are *classes* (not instances) on purpose:

    * ``dspy.make_signature``'s type guard accepts a class, so a Scale/Enum field
      survives signature rebuilds (``.insert`` / ``.append`` / ``as_arbiter``), unlike a
      bare annotation object which the guard rejects.
    * ``typing.get_origin()`` returns ``None`` for a plain class, so dspy's output parser
      stays on the pydantic path (``TypeAdapter(...).validate_python``) and applies our
      loose-coercion validator — whereas a real ``Literal`` would route to dspy's built-in
      Literal parser, which does no coercion.
    * ``__origin__ = Literal`` (a class attribute) makes dspy *render* the field as a
      Literal of the allowed choices.

    The metaclass only customizes ``repr`` so the annotation prints as
    ``modaic.Scale[1, 5]`` / ``modaic.Enum['A', 'B']`` instead of ``<class ...>``.
    """

    def __repr__(cls):
        return cls._modaic_repr


def _build(name: str, args: tuple, validate, modaic_type: str, modaic_args: list, repr_str: str):
    """Construct a Literal-like annotation class for ``args`` with ``validate`` coercion."""

    @classmethod
    def pydantic_core_schema(cls, source, handler):
        return core_schema.no_info_plain_validator_function(validate)

    @classmethod
    def pydantic_json_schema(cls, schema, handler):
        # Serialize as the underlying Literal (enum/const) so consumers that don't
        # understand the markers still get a valid schema, then tag it so a
        # serialize -> deserialize round-trip rebuilds the Scale/Enum rather than a
        # plain Literal. See modaic/serializers.py:json_to_type for the reconstruction.
        js = handler(core_schema.literal_schema(list(args)))
        js["x-modaic-type"] = modaic_type
        js["x-modaic-args"] = modaic_args
        return js

    return _LiteralLike(
        name,
        (),
        {
            "__origin__": Literal,
            "__args__": args,
            "_modaic_repr": repr_str,
            "__get_pydantic_core_schema__": pydantic_core_schema,
            "__get_pydantic_json_schema__": pydantic_json_schema,
        },
    )


@functools.lru_cache(maxsize=None)
def _enum(values: tuple):
    """Build (and memoize) the annotation class for ``Enum[*values]``."""

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

        if v in values:
            return v

        # Case-insensitive match — return the original-cased allowed value
        v_lower = v.lower()
        for a in values:
            if a.lower() == v_lower:
                return a

        raise ValueError(f"{v!r} is not one of {values!r}")

    rendered = ", ".join(repr(v) for v in values)
    return _build(
        name=f"Enum[{rendered}]",
        args=values,
        validate=validate,
        modaic_type="Enum",
        modaic_args=list(values),
        repr_str=f"modaic.Enum[{rendered}]",
    )


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
        return _enum(values)


@functools.lru_cache(maxsize=None)
def _scale(lo: int, hi: int):
    """Build (and memoize) the annotation class for ``Scale[lo, hi]``."""

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

    return _build(
        name=f"Scale[{lo}, {hi}]",
        args=tuple(range(lo, hi + 1)),
        validate=validate,
        modaic_type="Scale",
        modaic_args=[lo, hi],
        repr_str=f"modaic.Scale[{lo}, {hi}]",
    )


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
        return _scale(lo, hi)
