"""Infer the serving capabilities required by a DSPy signature."""

import re
import warnings
from collections.abc import Iterable, Mapping
from typing import Any

import dspy

from ..serializers import _DSPY_BASE_TYPE_SCHEMA_KEY, serialize_signature

_CANONICAL_MODALITY_ORDER = ("text", "image", "audio")
_DSPY_SCHEMA_MODALITIES = {
    "dspy.Image": "image",
    "dspy.Audio": "audio",
}
_DEFS_REF_PREFIX = "#/$defs/"
_MODALITY_RE = re.compile(r"^[a-z][a-z0-9_-]*$")
_SCHEMA_MAP_KEYS = frozenset({"$defs", "definitions", "properties", "patternProperties", "dependentSchemas"})
_SCHEMA_LIST_KEYS = frozenset({"allOf", "anyOf", "oneOf", "prefixItems"})
_SCHEMA_NODE_KEYS = frozenset(
    {
        "additionalProperties",
        "contains",
        "else",
        "if",
        "items",
        "not",
        "propertyNames",
        "then",
        "unevaluatedItems",
        "unevaluatedProperties",
    }
)


def _modality_values(modalities: Iterable[str] | str | None) -> set[str]:
    if modalities is None:
        return set()
    if isinstance(modalities, str):
        modalities = (modalities,)
    elif isinstance(modalities, Mapping) or not isinstance(modalities, Iterable):
        raise TypeError("modalities must be a string or iterable of strings")

    normalized = set()
    for modality in modalities:
        if not isinstance(modality, str):
            raise TypeError("modalities must contain only strings")
        value = modality.strip().lower()
        # Validate the wire format rather than a closed allowlist so newer
        # modalities remain forward-compatible with older SDK releases.
        if not _MODALITY_RE.fullmatch(value):
            raise ValueError(
                f"invalid modality {modality!r}; expected a name matching "
                "[a-z][a-z0-9_-]* after trimming and lowercasing"
            )
        normalized.add(value)
    return normalized


def canonicalize_modalities(*groups: Iterable[str] | str | None) -> list[str]:
    """Return deduplicated modality names in a stable metadata order."""
    modalities = {"text"}
    for group in groups:
        modalities.update(_modality_values(group))

    known = [name for name in _CANONICAL_MODALITY_ORDER if name in modalities]
    unknown = sorted(modalities.difference(_CANONICAL_MODALITY_ORDER))
    return [*known, *unknown]


def reconcile_explicit_modalities(
    stored_modalities: Iterable[str] | str | None,
    *,
    previous_inferred: Iterable[str] | str | None = None,
    previous_explicit: Iterable[str] | str | None = None,
) -> list[str]:
    """Recover explicit requirements from public metadata and provenance.

    Existing repositories without provenance conservatively treat every stored
    requirement as explicit. Once provenance is available, inferred
    requirements may disappear when a signature changes, while explicitly
    declared requirements remain until the caller resets stored metadata.
    """
    if previous_inferred is None or previous_explicit is None:
        return canonicalize_modalities(stored_modalities)

    if stored_modalities is None:
        # Once provenance exists, deleting the public metadata key is an
        # authoritative reset rather than an instruction to resurrect history.
        return ["text"]

    previous_explicit_values = _modality_values(previous_explicit)
    stored_values = _modality_values(stored_modalities)
    previous_inferred_values = _modality_values(previous_inferred)
    retained_explicit = previous_explicit_values.intersection(stored_values)
    newly_explicit = stored_values.difference(previous_inferred_values)
    return canonicalize_modalities(retained_explicit, newly_explicit)


def infer_required_modalities(
    signature: type[dspy.Signature],
    *,
    _warning_stacklevel: int = 2,
    _fallback_on_schema_error: bool = True,
) -> list[str]:
    """Infer input modalities from a signature's serialized JSON Schema.

    DSPy's custom schema generator gives Image and Audio unambiguous type
    markers. Walking the schema graph, rather than Python typing objects,
    naturally covers unions, containers, and nested Pydantic models.
    """
    try:
        schema = serialize_signature(signature)
    except Exception as exc:
        if not _fallback_on_schema_error:
            raise
        signature_name = getattr(signature, "__name__", None) or repr(signature)
        error_text = str(exc).strip()
        error_summary = error_text.splitlines()[0] if error_text else "<no detail>"
        warnings.warn(
            f"Could not infer input modalities for DSPy signature {signature_name}: "
            f"{type(exc).__name__}: {error_summary}. "
            "Falling back to ['text'] for automatic inference; explicitly "
            "declared metadata modalities are still preserved.",
            RuntimeWarning,
            stacklevel=_warning_stacklevel,
        )
        return ["text"]
    definitions = schema.get("$defs", {})
    detected = {"text"}
    visited_refs: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                walk(item)
            return
        if not isinstance(node, dict):
            return

        node_type = node.get("type")
        dspy_type = node.get(_DSPY_BASE_TYPE_SCHEMA_KEY, node_type)
        if isinstance(dspy_type, str) and (modality := _DSPY_SCHEMA_MODALITIES.get(dspy_type)):
            detected.add(modality)

        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith(_DEFS_REF_PREFIX) and ref not in visited_refs:
            visited_refs.add(ref)
            definition_name = ref.removeprefix(_DEFS_REF_PREFIX)
            walk(definitions.get(definition_name))

        for key in _SCHEMA_MAP_KEYS:
            children = node.get(key)
            if isinstance(children, dict):
                for child in children.values():
                    walk(child)
        for key in _SCHEMA_LIST_KEYS:
            children = node.get(key)
            if isinstance(children, list):
                walk(children)
        for key in _SCHEMA_NODE_KEYS:
            walk(node.get(key))

    for field_schema in schema.get("properties", {}).values():
        if field_schema.get("__dspy_field_type") == "input":
            walk(field_schema)

    return canonicalize_modalities(detected)


def merge_required_modalities(
    signature: type[dspy.Signature],
    *declared_groups: Iterable[str] | str | None,
    _warning_stacklevel: int = 3,
    _fallback_on_schema_error: bool = True,
) -> list[str]:
    """Merge declared requirements with inferred ones without downgrading."""
    return canonicalize_modalities(
        infer_required_modalities(
            signature,
            _warning_stacklevel=_warning_stacklevel,
            _fallback_on_schema_error=_fallback_on_schema_error,
        ),
        *declared_groups,
    )
