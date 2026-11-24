"""Normalized public HTTP contract used by the OpenAPI drift guard.

``build_snapshot`` reduces FastAPI's ``/openapi.json`` document to the parts
that form the public contract: paths, methods, request/response models, query
parameters, required fields and enum values. Titles, descriptions, operation
ids, examples, tags and server lists are dropped so the committed snapshot stays
small and does not churn on FastAPI/Pydantic patch releases.

The snapshot lives at ``tests/snapshots/openapi.json`` and is checked by
``tests/test_openapi_snapshot.py``. Regenerate it only for an intended API
change, then review the printed diff:

    UPDATE_OPENAPI=1 uv run pytest tests/test_openapi_snapshot.py
    uv run python scripts/update_openapi_snapshot.py

Both commands write the same file; neither talks to a network or a database.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

__all__ = [
    "DEFAULT_SNAPSHOT_PATH",
    "HTTP_METHODS",
    "PUBLIC_PATH_PREFIX",
    "SNAPSHOT_VERSION",
    "build_snapshot",
    "diff_snapshots",
    "load_snapshot",
    "write_snapshot",
]

#: Bumped only when the snapshot layout itself changes.
SNAPSHOT_VERSION = 1

#: Only the panel/API surface is guarded; FastAPI's own docs routes never show
#: up in ``paths`` but the prefix keeps the filter explicit if they ever do.
PUBLIC_PATH_PREFIX = "/api/"

#: Path item keys that are operations rather than path-level metadata.
HTTP_METHODS = frozenset({"delete", "get", "head", "options", "patch", "post", "put", "trace"})

#: Pydantic renders model references as JSON pointers into this section.
REF_PREFIX = "#/components/schemas/"

#: ``backend/`` — the path is resolved from this file, never from the cwd.
BACKEND_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_SNAPSHOT_PATH = BACKEND_ROOT / "tests" / "snapshots" / "openapi.json"


def build_snapshot(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Return the normalized public contract for one OpenAPI document.

    The result has two sections. ``operations`` maps ``"METHOD /path"`` to the
    request model, query parameters and per-status response signatures.
    ``models`` holds only the schemas reachable from those operations, with
    their sorted required fields, property type signatures and enum values.
    """
    paths = schema.get("paths")
    if not isinstance(paths, Mapping):
        raise TypeError("OpenAPI document has no 'paths' object")

    components = (schema.get("components") or {}).get("schemas") or {}
    if not isinstance(components, Mapping):
        raise TypeError("OpenAPI document has a malformed 'components.schemas'")

    operations: dict[str, Any] = {}
    roots: set[str] = set()
    for path in sorted(paths):
        if not path.startswith(PUBLIC_PATH_PREFIX):
            continue
        item = paths[path]
        if not isinstance(item, Mapping):
            continue
        for method in sorted(item):
            if method not in HTTP_METHODS:
                continue
            operation = item[method]
            if not isinstance(operation, Mapping):
                continue
            record, refs = _operation_record(operation)
            operations[f"{method.upper()} {path}"] = record
            roots.update(refs)

    models: dict[str, Any] = {}
    pending = sorted(roots)
    while pending:
        name = pending.pop(0)
        if name in models:
            continue
        raw = components.get(name)
        if not isinstance(raw, Mapping):
            raise TypeError(f"operation references unknown schema {name!r}")
        record, refs = _model_record(raw)
        models[name] = record
        pending.extend(sorted(refs - models.keys()))

    return {
        "schema": "scalping-public-api",
        "version": SNAPSHOT_VERSION,
        "path_prefix": PUBLIC_PATH_PREFIX,
        "operations": operations,
        "models": models,
    }


def _operation_record(operation: Mapping[str, Any]) -> tuple[dict[str, Any], set[str]]:
    """Flatten one operation into a comparable record plus the refs it uses."""
    refs: set[str] = set()

    body = _json_schema(operation.get("requestBody") or {})
    request = _type_signature(body) if body is not None else None
    refs |= _refs_in(body)

    query: dict[str, str] = {}
    parameters = operation.get("parameters")
    if isinstance(parameters, list):
        for parameter in parameters:
            if not isinstance(parameter, Mapping) or parameter.get("in") != "query":
                continue
            name = str(parameter.get("name"))
            value = _type_signature(parameter.get("schema"))
            query[name] = value + ("!" if parameter.get("required") else "")

    responses: dict[str, str] = {}
    raw_responses = operation.get("responses") or {}
    if isinstance(raw_responses, Mapping):
        for status in sorted(raw_responses):
            response = raw_responses[status]
            body = _json_schema(response) if isinstance(response, Mapping) else None
            responses[str(status)] = _type_signature(body) if body is not None else "none"
            refs |= _refs_in(body)

    record = {
        "request": request,
        "query": dict(sorted(query.items())),
        "responses": responses,
    }
    return record, refs


def _model_record(raw: Mapping[str, Any]) -> tuple[dict[str, Any], set[str]]:
    """Flatten one component schema into required fields, properties and enums."""
    properties: dict[str, str] = {}
    enums: dict[str, list[Any]] = {}
    refs: set[str] = set()

    raw_properties = raw.get("properties") or {}
    if isinstance(raw_properties, Mapping):
        for name in sorted(raw_properties):
            prop = raw_properties[name]
            properties[str(name)] = _type_signature(prop)
            refs |= _refs_in(prop)
            _collect_enums(prop, str(name), enums)

    required = raw.get("required") or []
    required_names = sorted(str(name) for name in required if str(name) in properties)

    record = {
        "required": required_names,
        "properties": properties,
    }
    if enums:
        record["enums"] = dict(sorted(enums.items()))
    return record, refs


def _json_schema(holder: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Return the ``application/json`` schema of a request body or response."""
    content = holder.get("content")
    if not isinstance(content, Mapping):
        return None
    media = content.get("application/json")
    if not isinstance(media, Mapping):
        return None
    schema = media.get("schema")
    return schema if isinstance(schema, Mapping) else None


def _type_signature(schema: Any) -> str:
    """Render a schema as a short, stable type string.

    References collapse to the model name so the ``models`` section owns the
    field detail; unions and nullable fields keep their members sorted.
    """
    if not isinstance(schema, Mapping):
        return "unknown"

    ref = schema.get("$ref")
    if isinstance(ref, str):
        return ref[len(REF_PREFIX) :] if ref.startswith(REF_PREFIX) else ref.rsplit("/", 1)[-1]

    for key in ("anyOf", "oneOf"):
        branches = schema.get(key)
        if isinstance(branches, list) and branches:
            members = {_type_signature(branch) for branch in branches}
            # ``null`` last keeps nullable fields readable: "string | null".
            return " | ".join(sorted(members, key=lambda part: (part == "null", part)))

    all_of = schema.get("allOf")
    if isinstance(all_of, list) and all_of:
        sequence = list(dict.fromkeys(_type_signature(branch) for branch in all_of))
        return sequence[0] if len(sequence) == 1 else " & ".join(sequence)

    enum = schema.get("enum")
    if isinstance(enum, list):
        return "enum[" + "|".join(str(value) for value in enum) + "]"

    type_ = schema.get("type")
    if isinstance(type_, list):
        return " | ".join(sorted(str(value) for value in type_))
    if type_ == "array":
        return f"array<{_type_signature(schema.get('items'))}>"
    if isinstance(type_, str):
        return type_
    if "properties" in schema or "additionalProperties" in schema:
        return "object"
    return "unknown"


def _refs_in(schema: Any) -> set[str]:
    """Collect every model name referenced anywhere inside one schema."""
    refs: set[str] = set()
    if isinstance(schema, Mapping):
        ref = schema.get("$ref")
        if isinstance(ref, str):
            refs.add(ref.removeprefix(REF_PREFIX))
            return refs
        for value in schema.values():
            refs |= _refs_in(value)
    elif isinstance(schema, list):
        for value in schema:
            refs |= _refs_in(value)
    return refs


def _collect_enums(schema: Any, prefix: str, out: dict[str, list[Any]]) -> None:
    """Record enum values under a dotted property path (``legs[].side``)."""
    if not isinstance(schema, Mapping):
        return
    if "$ref" in schema:
        # The referenced model records its own enums.
        return
    enum = schema.get("enum")
    if isinstance(enum, list):
        out[prefix] = list(enum)
        return
    for key in ("anyOf", "oneOf", "allOf"):
        branches = schema.get(key)
        if isinstance(branches, list):
            for branch in branches:
                _collect_enums(branch, prefix, out)
    items = schema.get("items")
    if isinstance(items, Mapping):
        _collect_enums(items, f"{prefix}[]", out)


def diff_snapshots(expected: Mapping[str, Any], current: Mapping[str, Any]) -> str:
    """Return a readable added/removed/changed report, or ``""`` when equal."""
    expected_operations = expected.get("operations") or {}
    current_operations = current.get("operations") or {}
    if not isinstance(expected_operations, Mapping) or not isinstance(current_operations, Mapping):
        raise TypeError("snapshot is missing its 'operations' object")

    lines: list[str] = []

    for key, label in (("schema", "snapshot name"), ("version", "snapshot version")):
        if expected.get(key) != current.get(key):
            lines.append(f"header: {label} {expected.get(key)!r} -> {current.get(key)!r}")

    added_ops = sorted(set(current_operations) - set(expected_operations))
    removed_ops = sorted(set(expected_operations) - set(current_operations))
    changed_ops = _changed(expected_operations, current_operations, _diff_operation)
    _render(lines, "operations", added_ops, removed_ops, changed_ops)

    expected_models = expected.get("models") or {}
    current_models = current.get("models") or {}
    if not isinstance(expected_models, Mapping) or not isinstance(current_models, Mapping):
        raise TypeError("snapshot is missing its 'models' object")

    added_models = sorted(set(current_models) - set(expected_models))
    removed_models = sorted(set(expected_models) - set(current_models))
    changed_models = _changed(expected_models, current_models, _diff_model)
    _render(lines, "models", added_models, removed_models, changed_models)

    return "\n".join(lines)


def _changed(
    expected: Mapping[str, Any],
    current: Mapping[str, Any],
    diff: Callable[[str, Any, Any], list[str]],
) -> list[tuple[str, list[str]]]:
    changed: list[tuple[str, list[str]]] = []
    for key in sorted(set(expected) & set(current)):
        details = diff(key, expected[key], current[key])
        if details:
            changed.append((key, details))
    return changed


def _render(
    lines: list[str],
    noun: str,
    added: Iterable[str],
    removed: Iterable[str],
    changed: list[tuple[str, list[str]]],
) -> None:
    added = list(added)
    removed = list(removed)
    total = len(added) + len(removed) + len(changed)
    if not total:
        return
    lines.append(f"{noun}: +{len(added)} -{len(removed)} ~{len(changed)}")
    for key in added:
        lines.append(f"  + {key}")
    for key in removed:
        lines.append(f"  - {key}")
    for key, details in changed:
        lines.append(f"  ~ {key}")
        lines.extend(f"      {detail}" for detail in details)


def _diff_operation(_key: str, expected: Any, current: Any) -> list[str]:
    if not isinstance(expected, Mapping) or not isinstance(current, Mapping):
        return ["record is malformed"]
    lines: list[str] = []

    if expected.get("request") != current.get("request"):
        lines.append(f"request: {_fmt(expected.get('request'))} -> {_fmt(current.get('request'))}")

    old_query = expected.get("query") or {}
    new_query = current.get("query") or {}
    for name in sorted(set(new_query) - set(old_query)):
        lines.append(f"+ query {name}: {new_query[name]}")
    for name in sorted(set(old_query) - set(new_query)):
        lines.append(f"- query {name}: {old_query[name]}")
    for name in sorted(set(old_query) & set(new_query)):
        if old_query[name] != new_query[name]:
            lines.append(f"~ query {name}: {old_query[name]} -> {new_query[name]}")

    old_responses = expected.get("responses") or {}
    new_responses = current.get("responses") or {}
    for status in sorted(set(new_responses) - set(old_responses)):
        lines.append(f"+ response {status}: {new_responses[status]}")
    for status in sorted(set(old_responses) - set(new_responses)):
        lines.append(f"- response {status}: {old_responses[status]}")
    for status in sorted(set(old_responses) & set(new_responses)):
        if old_responses[status] != new_responses[status]:
            lines.append(f"~ response {status}: {old_responses[status]} -> {new_responses[status]}")
    return lines


def _diff_model(_name: str, expected: Any, current: Any) -> list[str]:
    if not isinstance(expected, Mapping) or not isinstance(current, Mapping):
        return ["record is malformed"]
    lines: list[str] = []

    old_props = expected.get("properties") or {}
    new_props = current.get("properties") or {}
    for name in sorted(set(new_props) - set(old_props)):
        lines.append(f"+ field {name}: {new_props[name]}")
    for name in sorted(set(old_props) - set(new_props)):
        lines.append(f"- field {name}: {old_props[name]}")
    for name in sorted(set(old_props) & set(new_props)):
        old_value, new_value = old_props[name], new_props[name]
        if old_value == new_value:
            continue
        # Enum value changes are reported below with explicit +value/-value rows.
        if str(old_value).startswith("enum[") and str(new_value).startswith("enum["):
            continue
        lines.append(f"~ field {name}: {old_value} -> {new_value}")

    old_required = set(expected.get("required") or [])
    new_required = set(current.get("required") or [])
    for name in sorted(new_required - old_required):
        lines.append(f"+ required {name}")
    for name in sorted(old_required - new_required):
        lines.append(f"- required {name}")

    old_enums = expected.get("enums") or {}
    new_enums = current.get("enums") or {}
    for name in sorted(set(new_enums) - set(old_enums)):
        lines.append(f"+ enum {name}: {_enum_values(new_enums[name])}")
    for name in sorted(set(old_enums) - set(new_enums)):
        lines.append(f"- enum {name}: {_enum_values(old_enums[name])}")
    for name in sorted(set(old_enums) & set(new_enums)):
        old_values = [str(value) for value in old_enums[name]]
        new_values = [str(value) for value in new_enums[name]]
        if old_values == new_values:
            continue
        for value in new_values:
            if value not in old_values:
                lines.append(f"+ enum {name}: {value}")
        for value in old_values:
            if value not in new_values:
                lines.append(f"- enum {name}: {value}")
        before = _enum_values(old_values)
        after = _enum_values(new_values)
        lines.append(f"~ enum {name}: [{before}] -> [{after}]")
    return lines


def _enum_values(values: Any) -> str:
    return "|".join(str(value) for value in values)


def _fmt(value: Any) -> str:
    return "none" if value is None else str(value)


def load_snapshot(path: Path = DEFAULT_SNAPSHOT_PATH) -> dict[str, Any]:
    """Read the committed snapshot, naming the update command when it is absent."""
    if not path.exists():
        raise FileNotFoundError(
            f"missing OpenAPI snapshot {path}; create it with "
            "`UPDATE_OPENAPI=1 uv run pytest tests/test_openapi_snapshot.py`"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return data


def write_snapshot(path: Path, snapshot: Mapping[str, Any]) -> None:
    """Write the snapshot with sorted keys and one trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(snapshot, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    path.write_text(text, encoding="utf-8")
