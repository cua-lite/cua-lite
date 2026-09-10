"""Tool declaration: canonical schema, ``@tool`` compiler, and ``BaseTools``.

``BaseTools`` is the tool-layer authoring root for env-declared schemas carried
through ``metadata.extra_tool_schemas``. The canonical schema shape is the nested
function-tool dict this module builds.

Nested function-tool shape is canonical, here and on every persisted Lite path:
outer ``type`` plus ``function``, whose fields are ``name``, ``description``,
``parameters``, and optional ``strict``.
Provider- or model-specific projections are produced at their owning outlet,
not by moving those fields to a top-level schema shape.

Four separate questions live here; do not answer one with another's helper:

1. *What is canonical schema shape* -- ``LiteToolSchema``, ``make_tool_schema()``,
   ``tool_schema_name()``, ``tool_schema_parameters()``, and the
   ``validate_extra_tool_schemas()`` boundary.
2. *Compile a Python method into JSON Schema* -- the ``@tool`` decorator and the
   module-private compiler (``_compile_tool_method_schemas`` and its type-hint
   helpers), reached only from ``install_tool_schemas`` and the action-space
   compiler path.
3. *Route a possibly malformed call to an owner* -- the ``*_route_keys``
   predicates. Closed-key key-shape matching, deliberately stricter than JSON
   Schema, and never a validity answer. Owner-module only, off the
   ``lite.core.tools`` facade.
4. *Fully validate arguments* -- ``tool_schema_validation_errors()`` and
   ``tool_call_satisfies_schema()``, which defer to ``Draft202012Validator``
   and apply no route gate, plus ``schema_type_error_reason()``, which phrases
   one of those errors as model-visible feedback.

Usage:
    from lite.core.tools.schemas import BaseTools, install_tool_schemas, tool

    @install_tool_schemas
    class MyTools(BaseTools):
        @staticmethod
        @tool(text="Text to submit.")
        def say(text: str) -> dict[str, Any]:
            '''Say something.'''
            return make_tool_call("say", {"text": text})

Run:
    uv run pytest tests/core/tools/test_tool_schemas.py -q
"""

from __future__ import annotations

import copy
import inspect
import json
import types
from collections.abc import Callable
from functools import lru_cache
from typing import (
    Any,
    ClassVar,
    Literal,
    NotRequired,
    TypedDict,
    get_args,
    get_origin,
    get_type_hints,
)

from lite.core.errors import LiteContractError
from lite.core.tools.calls import make_tool_call, tool_call_arguments, tool_call_name


class LiteToolSchemaFunction(TypedDict):
    """Function declaration inside a canonical Lite tool schema."""

    name: str
    description: NotRequired[str]
    parameters: dict[str, Any]
    strict: NotRequired[bool]


class LiteToolSchema(TypedDict):
    """Canonical ``metadata.extra_tool_schemas[*]`` entry."""

    type: Literal["function"]
    function: LiteToolSchemaFunction


#: Every key a canonical Lite tool schema may carry. DERIVED from the two
#: TypedDicts above rather than re-listed, so the shape is declared exactly once:
#: a new field on ``LiteToolSchema`` is admitted by ``validate_extra_tool_schemas``
#: without an edit, and a field removed from it stops being admitted.
_LITE_TOOL_SCHEMA_OUTER_KEYS = frozenset(
    LiteToolSchema.__required_keys__ | LiteToolSchema.__optional_keys__
)
_LITE_TOOL_SCHEMA_FUNCTION_KEYS = frozenset(
    LiteToolSchemaFunction.__required_keys__
    | LiteToolSchemaFunction.__optional_keys__
)


def validate_extra_tool_schemas(
    schemas: list[dict[str, Any]],
    *,
    where: str = "metadata.extra_tool_schemas",
) -> None:
    """Validate the canonical ``LiteBaseMetadata.extra_tool_schemas`` invariant.

    This is a structural check for canonical Lite tool schemas, including the
    duplicate-name rule. It admits declared function members such as ``strict``
    without interpreting provider-specific schema semantics. ``where`` labels
    the validated surface in error messages.
    """
    if not isinstance(schemas, list):
        raise LiteContractError(f"{where} must be a list")
    seen: set[str] = set()
    for i, schema in enumerate(schemas):
        at = f"{where}[{i}]"
        if not isinstance(schema, dict):
            raise LiteContractError(f"{at} must be an object")
        extra_keys = set(schema) - _LITE_TOOL_SCHEMA_OUTER_KEYS
        if extra_keys:
            raise LiteContractError(f"{at} has noncanonical outer keys {sorted(extra_keys)}")
        if schema.get("type") != "function":
            raise LiteContractError(f"{at}.type must be 'function'")
        function = schema.get("function")
        if not isinstance(function, dict):
            raise LiteContractError(f"{at}.function must be an object")
        inner_extra_keys = set(function) - _LITE_TOOL_SCHEMA_FUNCTION_KEYS
        if inner_extra_keys:
            raise LiteContractError(
                f"{at}.function has noncanonical keys {sorted(inner_extra_keys)}"
            )
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise LiteContractError(f"{at}.function.name must be a non-empty string")
        if name in seen:
            raise LiteContractError(f"duplicate {where} name {name!r}")
        seen.add(name)
        if "description" in function and not isinstance(function["description"], str):
            raise LiteContractError(f"{at}.function.description must be a string")
        if "strict" in function and not isinstance(function["strict"], bool):
            raise LiteContractError(f"{at}.function.strict must be a bool")
        if not isinstance(function.get("parameters"), dict):
            raise LiteContractError(f"{at}.function.parameters must be an object")


def make_tool_schema(
    name: str,
    *,
    description: str = "",
    parameters: dict[str, Any] | None = None,
    strict: bool | None = None,
) -> dict[str, Any]:
    """Build a canonical Lite tool schema with optional function-level ``strict``."""
    function: dict[str, Any] = {
        "name": name,
        "description": description,
        "parameters": (
            parameters
            if parameters is not None
            else {"type": "object", "properties": {}, "required": []}
        ),
    }
    if strict is not None:
        function["strict"] = strict
    return {
        "type": "function",
        "function": function,
    }


def _canonical_function_field(schema: dict[str, Any], field: str) -> Any:
    """Read one field out of the canonical nested ``function`` envelope.

    These accessors are the tool-schema boundary: callers reach them with
    schemas that may still carry a provider-shaped flat spelling, where
    ``name``/``parameters`` sit at the top level. That shape fails here with the
    named Lite contract error listing the noncanonical outer keys, not a bare
    ``KeyError`` a caller cannot attribute.
    """
    if not isinstance(schema, dict):
        raise LiteContractError(f"tool schema must be an object, got {type(schema).__name__}")
    function = schema.get("function")
    if not isinstance(function, dict) or field not in function:
        outer_keys = sorted(set(schema) - _LITE_TOOL_SCHEMA_OUTER_KEYS)
        raise LiteContractError(
            f"tool schema has no canonical function.{field}: expected it nested "
            f"under 'function', found noncanonical outer keys {outer_keys}"
        )
    return function[field]


def tool_schema_name(schema: dict[str, Any]) -> str:
    """Return the canonical Lite tool schema's function name."""
    return _canonical_function_field(schema, "name")


def tool_schema_parameters(schema: dict[str, Any]) -> dict[str, Any]:
    """Return the canonical Lite tool schema's JSON-schema parameters."""
    return _canonical_function_field(schema, "parameters")


# Key-route predicates. These stay in the schema owner module, not the
# ``lite.core.tools`` facade: callers use them to choose an execution owner
# before full env/schema validation, not to validate a tool call.
def tool_arguments_match_schema_route_keys(
    schema: dict[str, Any],
    arguments: dict[str, Any],
) -> bool:
    """Return whether ``arguments`` are route-compatible with schema keys.

    This intentionally does not answer full JSON Schema validity. Route-
    compatible means required keys are present and the argument key set is
    closed unless ``additionalProperties`` is literally ``true``. Schema-valued
    ``additionalProperties`` remains closed here so colliding tool names do not
    widen admission before full validation.
    """
    try:
        parameters = tool_schema_parameters(schema)
    except LiteContractError:
        return False
    if not isinstance(parameters, dict):
        return False
    if not isinstance(arguments, dict):
        return False

    required = parameters.get("required", [])
    if not isinstance(required, list) or any(
        not isinstance(name, str) for name in required
    ):
        return False
    if any(name not in arguments for name in required):
        return False

    properties = parameters.get("properties")
    if not isinstance(properties, dict):
        return True

    property_names = {name for name in properties if isinstance(name, str)}
    if parameters.get("additionalProperties") is True:
        return True
    return set(arguments).issubset(property_names)


def tool_name_and_arguments_match_schema_route_keys(
    *,
    name: str,
    arguments: dict[str, Any],
    schema: dict[str, Any],
) -> bool:
    """Route a named argument set to a Lite schema by name and key shape.

    Closed-key semantics, exactly as in ``tool_arguments_match_schema_route_keys``.
    This is not schema satisfaction: use ``tool_call_satisfies_schema()`` for
    that.
    """
    try:
        if tool_schema_name(schema) != name:
            return False
    except LiteContractError:
        return False
    if not isinstance(arguments, dict):
        return False
    return tool_arguments_match_schema_route_keys(schema, arguments)


def tool_call_matches_schema_route_keys(
    call: dict[str, Any],
    schema: dict[str, Any],
) -> bool:
    """Route a canonical Lite tool call to a schema by name and key shape.

    Closed-key semantics, exactly as in ``tool_arguments_match_schema_route_keys``.
    A call that is not canonically shaped routes nowhere and returns ``False``.
    """
    try:
        name = tool_call_name(call)
        arguments = tool_call_arguments(call)
    except (KeyError, TypeError):
        return False
    return tool_name_and_arguments_match_schema_route_keys(
        name=name,
        arguments=arguments,
        schema=schema,
    )


@lru_cache(maxsize=512)
def _validator_for(parameters_json: str) -> Any:
    """Build (and metaschema-check) the validator for one parameters schema.

    Keyed on the serialized schema because that is what decides the validator:
    two calls with equal parameters get the same object. Only ``iter_errors`` is
    per-arguments work — ``check_schema`` walks the Draft 2020-12 metaschema and
    the constructor compiles ``$ref`` resolution, and a staging run would otherwise
    repeat both once per action call over a handful of distinct schemas.

    The key is ``json.dumps`` WITHOUT ``sort_keys``, deliberately. Sorting would
    rebuild every dict in the schema in alphabetical order, and ``iter_errors``
    emits in schema order — so with the stable sort in the caller, two errors on
    the same path would swap and the MODEL-VISIBLE reason would change (a bad
    ``Literal`` arg reported "must be one of [...]" instead of "must be a
    string"). Two equal schemas written in different key order merely cost one
    extra cache entry, which is the cheap side of that trade.
    """
    from jsonschema import Draft202012Validator

    parameters = json.loads(parameters_json)
    Draft202012Validator.check_schema(parameters)
    return Draft202012Validator(parameters)


def tool_schema_validation_errors(
    schema: dict[str, Any],
    arguments: dict[str, Any],
) -> list[Any]:
    """Return sorted JSON-schema validation errors for tool arguments."""
    parameters = tool_schema_parameters(schema)
    validator = _validator_for(json.dumps(parameters))
    return sorted(
        validator.iter_errors(arguments),
        key=lambda err: list(err.path),
    )


def schema_type_error_reason(field: str, expected: str) -> str:
    """Model-visible reason for one JSON-schema ``type`` mismatch.

    Owned here because ``tool_schema_validation_errors()`` above produces the
    errors this phrases. Callers derive ``field`` themselves -- the label differs
    by boundary (``computer.arguments.actions[0].text`` inside an action batch,
    ``answer.arguments.text`` for a standalone extra tool) -- and share only the
    article-plus-type wording, so one mismatch reads the same at every boundary.
    """
    article = "an" if expected[:1].lower() in "aeiou" else "a"
    return f"{field} must be {article} {expected}"


def tool_call_satisfies_schema(
    call: dict[str, Any],
    schema: dict[str, Any],
) -> bool:
    """True when a Lite tool call's name and arguments satisfy a JSON schema.

    Schema SATISFACTION, not closed-key routing: after the tool name matches,
    this defers entirely to ``Draft202012Validator``. It deliberately does not
    run the ``*_route_keys`` gate first, so a schema that opens itself with a
    schema-valued ``additionalProperties`` gets the answer its own JSON Schema
    declares rather than the stricter one routing wants.
    """
    try:
        if tool_call_name(call) != tool_schema_name(schema):
            return False
        arguments = tool_call_arguments(call)
    except (KeyError, TypeError, LiteContractError):
        return False
    if not isinstance(arguments, dict):
        return False
    return not tool_schema_validation_errors(schema, arguments)


def tool(**descriptions: str) -> Callable:
    """Mark a static tool method for automatic Lite schema generation."""

    def decorator(func: Callable) -> Callable:
        func._tool_descriptions = descriptions
        return func

    return decorator


def _compile_tool_method_schemas(
    cls: type,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Generate tool schemas/default maps from ``@tool`` methods on ``cls``."""
    schemas: dict[str, dict[str, Any]] = {}
    defaults_by_name: dict[str, dict[str, Any]] = {}

    for name, attr in cls.__dict__.items():
        method = attr.__func__ if isinstance(attr, staticmethod) else attr

        if not (callable(method) and hasattr(method, "_tool_descriptions")):
            continue

        descriptions: dict[str, str] = method._tool_descriptions
        sig = inspect.signature(method)
        hints = get_type_hints(method)

        properties = {}
        required = []
        defaults = {}

        for param_name, param in sig.parameters.items():
            if param_name in ("return", "self", "cls"):
                continue

            py_type = hints.get(param_name, str)
            prop = _python_type_to_json_schema(py_type)

            if param_name in descriptions:
                prop["description"] = descriptions[param_name]

            properties[param_name] = prop

            has_default = param.default is not inspect.Parameter.empty
            is_optional = _is_optional_type(py_type)

            if not has_default and not is_optional:
                required.append(param_name)

            if has_default and param.default is not None:
                defaults[param_name] = param.default

        schemas[name] = make_tool_schema(
            name,
            description=inspect.cleandoc(method.__doc__ or ""),
            parameters={
                "type": "object",
                "properties": properties,
                "required": required,
            },
        )

        if defaults:
            defaults_by_name[name] = defaults

    return schemas, defaults_by_name


def _python_type_to_json_schema(py_type: Any) -> dict[str, Any]:
    """Convert a Python type hint to a JSON Schema property definition."""
    origin = get_origin(py_type)
    args = get_args(py_type)

    if origin is types.UnionType or (
        hasattr(origin, "__name__") and origin.__name__ == "Union"
    ):
        non_none_args = [a for a in args if a is not type(None)]
        if non_none_args:
            return _python_type_to_json_schema(non_none_args[0])

    if origin is Literal:
        enum_values = list(args)
        first_val = enum_values[0]
        if isinstance(first_val, int):
            return {"type": "integer", "enum": enum_values}
        if isinstance(first_val, str):
            return {"type": "string", "enum": enum_values}
        return {"enum": enum_values}

    if origin is list:
        if args:
            return {"type": "array", "items": _python_type_to_json_schema(args[0])}
        return {"type": "array"}

    type_map = {str: "string", int: "integer", float: "number", bool: "boolean"}
    return {"type": type_map.get(py_type, "string")}


def _is_optional_type(py_type: Any) -> bool:
    """Check if a type hint is optional."""
    origin = get_origin(py_type)
    if origin is type(None):
        return True
    args = get_args(py_type)
    return bool(args and type(None) in args)


class BaseTools:
    """The TOOL-layer root: schemas plus the accessors for emitted tool catalogs.

    Carries no registry key, conversion policy, or action layer. The emitted
    catalog is read through ``get_tool_schemas()``, ``get_tool_names()``, and
    ``get_tool_schema(name)``.
    """

    _SCHEMAS: ClassVar[dict[str, dict[str, Any]]] = {}
    _DEFAULTS: ClassVar[dict[str, dict[str, Any]]] = {}

    @staticmethod
    def _make_call(
        name: str,
        defaults: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build one canonical Lite tool call, dropping omitted/default-valued args.

        Action-space roots bind this body so tool-call construction uses one
        policy.
        """
        defaults = defaults or {}
        arguments = {
            key: value
            for key, value in kwargs.items()
            if value is not None and (key not in defaults or value != defaults[key])
        }
        return make_tool_call(name, arguments)

    @classmethod
    def get_tool_schemas(cls, include: list[str] | None = None) -> list[dict[str, Any]]:
        """Every schema this set EMITS, or the subset ``include`` SELECTS.

        On one-schema-per-tool sets, ``include`` selects emitted tool names and
        unknown names raise. Action-batch sets override this with action-layer
        filter semantics. Result order follows declaration order and repeats
        collapse.

        Raises:
            LiteContractError: a name in ``include`` this set does not declare.
        """
        schemas = list(cls._SCHEMAS.values())
        if include is not None:
            include_set = set(include)
            declared = {tool_schema_name(schema) for schema in schemas}
            unknown = sorted(include_set - declared)
            if unknown:
                raise LiteContractError(
                    f"unknown {cls.__name__} tool names in include=: {unknown}; "
                    f"available: {sorted(declared)}"
                )
            schemas = [schema for schema in schemas if tool_schema_name(schema) in include_set]
        return copy.deepcopy(schemas)

    @classmethod
    def get_tool_names(cls) -> frozenset[str]:
        """Names of the tools this set EMITS — derived from ``get_tool_schemas()``.

        ``get_tool_schemas()`` is the single root of the tool layer; the other
        two accessors are derived from it so both tool-layer invariants
        (``len(names) == len(schemas)`` and ``get_tool_schema(n) is not None
        <=> n in get_tool_names()``) hold by construction, including on
        action-batch sets whose whole surface collapses into one emitted tool.
        """
        return frozenset(tool_schema_name(schema) for schema in cls.get_tool_schemas())

    @classmethod
    def get_tool_schema(cls, name: str) -> dict[str, Any] | None:
        """One EMITTED tool schema by name — derived from ``get_tool_schemas()``.

        Scans the unfiltered emitted-schema list because this getter is
        miss-tolerant while ``include`` selection raises on unknown names.
        """
        for schema in cls.get_tool_schemas():
            if tool_schema_name(schema) == name:
                return schema
        return None


def install_tool_schemas(cls: type[BaseTools]) -> type[BaseTools]:
    """Compile ``cls``'s own ``@tool`` methods into its schema/default tables."""
    schemas, defaults = _compile_tool_method_schemas(cls)
    cls._SCHEMAS = schemas
    cls._DEFAULTS = defaults
    return cls


__all__ = [
    "BaseTools",
    "LiteToolSchema",
    "LiteToolSchemaFunction",
    "install_tool_schemas",
    "make_tool_schema",
    "schema_type_error_reason",
    "tool",
    "tool_call_satisfies_schema",
    "tool_schema_name",
    "tool_schema_parameters",
    "tool_schema_validation_errors",
    "validate_extra_tool_schemas",
]
