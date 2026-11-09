"""Validate the native datetime primitive without inventing partial dates."""

import calendar
import re
from typing import Any


def parse_datetime(value: Any, pattern: str) -> str:
    if not isinstance(value, str):
        raise ValueError("Datetime answers must be strings")
    expression = re.escape(pattern)
    fields = {"YYYY": "year", "MM": "month", "DD": "day", "HH": "hour", "SS": "second"}
    # MM means minutes after T or in a time-only pattern, months otherwise.
    if "T" in pattern:
        left, right = pattern.split("T")
        parts = [(left, False), (right, True)]
    else:
        parts = [(pattern, pattern.startswith("HH"))]
    expressions = []
    for part, clock in parts:
        expression = re.escape(part)
        for token, field in fields.items():
            name = "minute" if token == "MM" and clock else field
            expression = expression.replace(token, rf"(?P<{name}>[0-9]{{{len(token)}}})")
        expressions.append(expression)
    match = re.fullmatch("T".join(expressions), value)
    if match is None:
        raise ValueError(f"Datetime answer does not match {pattern}")
    values = {key: int(v) for key, v in match.groupdict().items() if v is not None}
    if "month" in values and not 1 <= values["month"] <= 12:
        raise ValueError("Invalid month")
    if (
        "day" in values
        and not 1 <= values["day"] <= calendar.monthrange(values["year"], values["month"])[1]
    ):
        raise ValueError("Invalid day")
    for key, limit in (("hour", 24), ("minute", 60), ("second", 60)):
        if values.get(key, 0) >= limit:
            raise ValueError(f"Invalid {key}")
    return value
