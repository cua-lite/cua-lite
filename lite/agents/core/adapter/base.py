"""Provider-independent adapter interfaces for local/HF-style agents.

Adapters are the boundary between canonical Lite trajectories and a model
family's wire format. The public path is:

1. resolve an adapter from :class:`AgentAdapterRegistry`;
2. normalize construction kwargs and env metadata in ``__post_init__``;
3. render one turn with ``render_step`` and model-specific prompt text;
4. parse model output back into a canonical Lite assistant message;
5. convert canonical tool feedback into the next model-visible turn.

Concrete adapters override ``_process_image_after_target`` (per-image
transform after any configured target resize), ``render_step`` (per-turn
prompt/message rendering), and the message conversion hooks. The base class
owns shared tool-surface validation, raw-response replay, action-batch
feedback, and the ``unroll`` skeleton that combines turns into an
:class:`AgentSample` (``processed_images`` / ``steps`` / ``metadata``).

Usage:
    @dataclasses.dataclass
    class MyAdapter(BaseAgentAdapter, key="my_agent@desktop@use"):
        action_space: MyActionSpace = dataclasses.field(default_factory=MyActionSpace)
        protocol: BaseProtocol = dataclasses.field(default_factory=FullHistoryProtocol)

        def render_step(self, sample, k, processed):
            ...

        def _convert_message_to_agent(self, message, **kwargs):
            ...

    adapter = AgentAdapterRegistry.get(
        "my_agent@desktop@use",
        metadata=LiteCUAMetadata(dims=("desktop", "use")),
    )
    agent_sample = adapter.unroll(lite_sample)
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import logging
import re
import typing
from abc import ABC, abstractmethod
from typing import Any

from PIL import Image

import lite.core.messages.image_refs as _image_refs
from lite.agents.core.action_space import BaseActionSpace
from lite.agents.core.action_space.base import (
    assemble_tool_schemas,
)
from lite.agents.core.protocol import BaseProtocol
from lite.agents.types import AgentMessage, AgentSample, AgentStep
from lite.core import (
    LiteMessage,
    LiteSample,
    LiteToolResult,
)
from lite.core.messages.content import ASSISTANT_ROLE, TEXT_PART
from lite.core.messages.final import is_canonical_no_tool_call_final_message
from lite.core.messages.turns import count_sample_turns
from lite.core.metadata import LiteBaseMetadata, LiteCUAMetadata
from lite.core.tools.action_space import (
    LITE_ACTION_BATCH_TOOL_NAMES,
    lite_action_batch_child_name_errors,
    validate_lite_action_batch_structure,
)
from lite.core.tools.calls import (
    make_tool_call,
    tool_call_arguments,
    tool_call_name,
)
from lite.core.tools.extra_tools import (
    active_extra_tool_names as _active_extra_tool_names,
)
from lite.core.tools.extra_tools import (
    extra_tool_name_and_arguments_are_admitted,
)
from lite.core.tools.schemas import (
    tool_name_and_arguments_match_schema_route_keys,
    tool_schema_name,
)
from lite.utils.image import normalize_resolution
from lite.utils.registry import BaseRegistry, RegistryKeyed

AdapterImageCarrier = Image.Image | str


def _resize_to_target(img: Image.Image, target: tuple[int, int]) -> Image.Image:
    """Resize to exactly ``target`` dimensions (stretch, non-AR-preserving).

    ``agent_kwargs.resolution`` is the literal output size the model operates
    in: set ``(W, H)`` → image becomes ``W×H`` regardless of source AR. The
    caller is responsible for picking a target whose AR matches the env to
    avoid distortion (e.g. 1920×1080 16:9 envs → use ``(1600, 900)`` or
    ``(1920, 1080)``, not ``(1024, 768)`` which would squash to 4:3).

    Identity if source already matches ``target`` exactly.
    """
    if img.size == target:
        return img
    return img.resize(target, Image.LANCZOS)


# ``AsIsAdapter`` defaults for the canonical cua-lite dialect. The factories
# return fresh core defaults for each adapter instance.
def _default_lite_action_space() -> BaseActionSpace:
    from lite.agents.core.action_space.base import LiteDesktopActionSpace

    return LiteDesktopActionSpace()


def _default_full_history_protocol() -> BaseProtocol:
    from lite.agents.core.protocol.base import FullHistoryProtocol

    return FullHistoryProtocol()


def _adapter_construction_default_metadata(adapter: BaseAgentAdapter) -> LiteCUAMetadata:
    """Default metadata for direct adapter construction.

    This is intentionally small: the adapter base mirrors the action-space
    platform so direct unit construction has a usable prompt surface, but it
    does not infer task type from registry-key text. Non-``use`` adapters that
    need metadata without an env/sample must declare that default on the
    subclass or receive explicit ``metadata=LiteCUAMetadata(...)`` from the caller.

    An action-space platform outside :class:`LiteCUAMetadata.Platform` raises here
    rather than being coerced to a default: the platform decides the whole
    prompt/tool surface, so a silent fallback would render the wrong one.
    """
    return LiteCUAMetadata(
        dims=(
            LiteCUAMetadata.Platform(adapter.action_space.platform).value,
            LiteCUAMetadata.TaskType.USE.value,
        ),
    )


logger = logging.getLogger(__name__)

TOOL_SURFACE_KWARGS = frozenset(
    {
        "extra_tools",
        "extra_tool_schemas",
        "valid_actions",
        "others",
    }
)
AGENT_KWARGS_TOOL_SURFACE_KEYS = TOOL_SURFACE_KWARGS | {"metadata"}


def tool_surface_agent_kwarg_names(agent_kwargs: dict[str, Any] | None) -> frozenset[str]:
    """Return env/tool-surface keys that must not be loose agent kwargs."""
    return AGENT_KWARGS_TOOL_SURFACE_KEYS & set(agent_kwargs or {})


@functools.cache
def _warn_adapter_key_mismatch(expected: str | None, actual: str | None) -> None:
    """Warn once per ``(expected, actual)`` pair (lru_cache = dedup)."""
    logger.warning(
        "raw_response present but adapter_key mismatch (expected %r, got %r); "
        "falling back to canonical conversion",
        expected,
        actual,
    )


# The log is this feedback's ONLY surface, deliberately: the env already answered
# the model with the same wording when it took the step, and the canonical call is
# contractually preserved intact, so re-detecting it during render tells no runtime
# consumer anything it can act on. Returning it from the render path would add a
# channel nothing reads. Dedup is process-global because a trajectory is re-rendered
# every turn; without it one rejected child warns once per turn for the whole run.
# ``tests/agents/core/adapter/test_render_invalid_batch_child.py`` reads the
# wording off caplog and calls ``.cache_clear()`` first — dropping this decorator
# fails loudly, not silently.
@functools.cache
def _warn_invalid_action_batch_child(adapter_name: str, feedback: str) -> None:
    """Warn once per ``(adapter, feedback)`` pair (lru_cache = dedup)."""
    logger.warning(
        "%s is rendering a canonical action-batch the env already rejected: %s",
        adapter_name,
        feedback,
    )


# =============================================================================
# Public API: Registry
# =============================================================================


class AgentAdapterRegistry(BaseRegistry["BaseAgentAdapter"], cache_by_default=False):
    """Registry for agent adapter classes.

    Extra constructor kwargs are auto-packed into the adapter's ``kwargs`` dict,
    which ``_apply_kwargs()`` consumes as protocol or adapter-field settings
    and rejects if unknown (see ``BaseRegistry.get()``). Tool-surface data
    travels through ``metadata=LiteCUAMetadata(...)``, not loose adapter kwargs.

    Example:
        AgentAdapterRegistry.get(
            "my_agent@desktop@use",
            metadata=LiteCUAMetadata(
                dims=("desktop", "use"),
                extra_tool_schemas=[...],
            ),
            protocol_kwargs={"full_history_size": 4},
        )
    """

    @classmethod
    def raw_response_key_matches(
        cls,
        stamped_key: str | None,
        adapter_key: str | None,
    ) -> bool:
        """Return whether a raw-response sidecar belongs to ``adapter_key``.

        Exact keys match directly. Regex matching is only enabled for strings
        registered as adapter patterns in this registry, in either direction:
        a registered pattern stamp can match a concrete adapter key, and a
        registered pattern adapter key can match a concrete stamp. Arbitrary
        regex-looking sidecar or target strings do not authorize replay.

        ``list_patterns()`` returns only strings the registry already compiled
        at registration, so ``re.fullmatch`` here cannot see an invalid pattern.
        """
        if not isinstance(stamped_key, str) or not isinstance(adapter_key, str):
            return False
        if stamped_key == adapter_key:
            return True

        registered_patterns = set(cls.list_patterns())
        if stamped_key in registered_patterns and re.fullmatch(stamped_key, adapter_key):
            return True
        if adapter_key in registered_patterns and re.fullmatch(adapter_key, stamped_key):
            return True
        return False


# =============================================================================
# Base Agent Adapter Class
# =============================================================================


@dataclasses.dataclass
class BaseAgentAdapter(RegistryKeyed, ABC):
    """Base class for adapting CUA-Lite messages to model-family messages.

    The hot path has five responsibilities:

    - construction-time metadata and agent-kwarg normalization;
    - prompt and image rendering for the current turn;
    - model-output parsing into canonical Lite messages;
    - action-space conversion with active env extra tools;
    - replay of raw provider responses only when the adapter key matches.

    A concrete adapter handles:
    - Converting a :class:`LiteSample` trajectory to an :class:`AgentSample`
      via :meth:`unroll` (skeleton; do NOT override).
    - Per-image processing (e.g. resize) via
      :meth:`_process_image_after_target`, after the base target-resize step.
    - Per-turn rendering via :meth:`render_step` hook.
    - Per-message helpers (:meth:`_convert_message_to_agent`,
      :meth:`convert_message_from_agent`,
      :meth:`parse_raw_assistant_response`) used inside ``render_step``
      and at message boundaries between agent and CUA-lite.
    - Action space conversions (delegated to ``self.action_space``).
    - Protocol-driven history shaping (delegated to ``self.protocol``).

    Attributes:
        action_space: The action space for this adapter.
        protocol: The interaction protocol.
        system_prompt: Optional system prompt template.
        kwargs: Registry-packed construction kwargs; consumed or rejected in
            ``__post_init__`` via ``_apply_kwargs()``.
    """

    # -- Core components --
    action_space: BaseActionSpace
    protocol: BaseProtocol

    # -- Configuration --
    system_prompt: str | None = None
    # Canonical (width, height) the adapter wants the model to operate in.
    # Applied as an EXACT stretch (non-AR-preserving) BEFORE adapter-specific
    # transforms (smart_resize / linear_resize / etc). Set ``(W, H)`` → image
    # becomes ``W×H`` exactly. Pick a target whose AR matches the env to
    # avoid distortion. ``None`` = identity (no target-resize step). YAML
    # knob: ``agent_kwargs.resolution``.
    resolution: tuple[int, int] | None = None

    # -- Env metadata (single source of truth) --
    # Forwarded by ``lite.agents.factory.make``. Subclass code reads
    # env-specific hints DIRECTLY from this object:
    #
    #   self.metadata.extra_tool_schemas        # list[dict], always populated
    #   self.metadata.valid_actions      # list[str] | None, CUA metadata only
    #   self.metadata.others["<key>"]    # env-specific hints (e.g. "apps")
    #
    # Env is the authoritative source — yaml configures the env
    # (env_kwargs), and the env derives its own metadata from that config.
    # ``__post_init__`` below installs a default so subclass code never has
    # to guard against ``None``.
    metadata: LiteBaseMetadata | None = None

    # -- Registry-packed construction kwargs consumed in __post_init__ --
    kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)

    # -- Class-level metadata --
    _registry = AgentAdapterRegistry

    def _apply_kwargs(self) -> None:
        """Consume registry-packed kwargs into protocol or adapter fields.

        Special keys (consumed first):
        - ``protocol_key``: replace ``self.protocol`` with a new instance
          resolved from :class:`ProtocolRegistry` by this key.
        - ``protocol_kwargs``: dict of attributes applied to the (possibly
          new) protocol. Explicit and unambiguous — preferred over relying
          on the implicit ``hasattr(self.protocol, k)`` fallback.

        Remaining kwargs are routed to declared adapter dataclass fields only.
        Protocol attributes must go through ``protocol_kwargs``. Unknown keys
        fail here because callers reach adapter construction through this
        registry-packed boundary, and a typo should not silently leave the
        adapter on defaults.

        Applying ``protocol_kwargs`` re-runs the protocol's ``__post_init__``,
        because ``setattr`` on an already-constructed protocol instance does
        not re-enter it.
        """
        if not isinstance(self.kwargs, dict):
            raise TypeError(
                f"{type(self).__name__}: kwargs must be a dict, "
                f"got {type(self.kwargs).__name__}"
            )
        if not self.kwargs:
            return
        remaining = dict(self.kwargs)

        # 1. Swap protocol if requested.
        protocol_key = remaining.pop("protocol_key", None)
        if protocol_key is not None:
            from lite.agents.core.protocol import ProtocolRegistry

            self.protocol = ProtocolRegistry.get(protocol_key)

        # 2. Apply explicit protocol_kwargs.
        #
        # ``v is None`` is APPLIED, not skipped: several protocols spell
        # "unbounded" as ``None`` for their history-size field, so skipping it
        # would make a YAML ``null`` a silent no-op -- the one spelling a
        # config has for that value.
        protocol_kwargs = remaining.pop("protocol_kwargs", None)
        if protocol_kwargs is not None:
            if not isinstance(protocol_kwargs, dict):
                raise TypeError(
                    f"{type(self).__name__}: protocol_kwargs must be a dict, "
                    f"got {type(protocol_kwargs).__name__}"
                )
            protocol_fields = (
                {field.name for field in dataclasses.fields(self.protocol)}
                if dataclasses.is_dataclass(self.protocol)
                else set(vars(self.protocol))
            )
            unknown_protocol_kwargs = sorted(
                k for k in protocol_kwargs if k not in protocol_fields
            )
            if unknown_protocol_kwargs:
                raise TypeError(
                    f"{type(self).__name__}: protocol "
                    f"{type(self.protocol).__name__} got unknown protocol_kwargs "
                    f"{unknown_protocol_kwargs}. Available fields: "
                    f"{sorted(protocol_fields)}"
                )
            for k, v in protocol_kwargs.items():
                setattr(self.protocol, k, v)
            # Re-run the protocol's own validation: construct-then-setattr
            # never re-enters ``__post_init__``.
            # The ``getattr`` is load-bearing -- most registered protocols define
            # no ``__post_init__``.
            post_init = getattr(self.protocol, "__post_init__", None)
            if post_init is not None:
                post_init()

        # 3. Route remaining kwargs to adapter fields only.
        #
        # ``v is None`` is APPLIED here too, for the same reason step 2 gives, and
        # so that a ``None`` reaches ``unrouted.append(k)`` below: otherwise
        # ``extra_tools: null`` slips past the ``TOOL_SURFACE_KWARGS`` check and is
        # silently dropped instead of hard-failing.
        unrouted = []
        adapter_fields = {field.name for field in dataclasses.fields(self)}
        for k, v in list(remaining.items()):
            if k in adapter_fields and k != "kwargs":
                setattr(self, k, v)
                remaining.pop(k)
            else:
                unrouted.append(k)
        # Hard-fail on tool-surface settings passed as LOOSE kwargs. Metadata
        # fields belong on the ``metadata`` object; ``extra_tools`` is an env
        # surface selector that must already have been resolved into
        # metadata.extra_tool_schemas before adapter construction. Routing any
        # of these here would silently drop them (they aren't adapter fields),
        # rendering with the wrong tool surface.
        misrouted = TOOL_SURFACE_KWARGS & set(unrouted)
        if misrouted:
            raise TypeError(
                f"{type(self).__name__}: {sorted(misrouted)} are tool-surface "
                f"settings, not adapter kwargs — pass resolved surface via "
                f"metadata=LiteCUAMetadata(...), not as loose kwargs (they would "
                f"be silently dropped otherwise)."
            )
        if unrouted:
            raise TypeError(
                f"{type(self).__name__}: unknown adapter kwargs "
                f"{sorted(unrouted)}. Available fields: "
                f"{sorted(adapter_fields - {'kwargs'})}"
            )
        self.kwargs = remaining

    def __post_init__(self):
        # This order is behavior, not cosmetic:
        # metadata default -> kwargs -> resolution normalization ->
        # action-space type check -> extra-tool collision check.
        # Ensure ``self.metadata`` is always populated so subclass code
        # can read ``self.metadata.{extra_tool_schemas,valid_actions,others}``
        # unconditionally. Factory/rollout construction supplies the env's
        # real metadata; this default catches direct construction (e.g. unit tests).
        if self.metadata is None:
            self.metadata = _adapter_construction_default_metadata(self)
        self._apply_kwargs()
        # Make the declared ``tuple[int, int] | None`` true: ``_apply_kwargs``
        # setattr's ``agent_kwargs.resolution`` straight from config, where a
        # pair is commonly a list. Every construction path lands here, so this
        # is the one place it has to happen — and NOT at the ``img.size ==
        # target`` comparison in :func:`_resize_to_target`, which would leave
        # the type lie in place for every other reader.
        self.resolution = normalize_resolution(self.resolution)
        hints = typing.get_type_hints(type(self))
        expected = hints.get("action_space")
        if expected is not None and isinstance(expected, type):
            if not isinstance(self.action_space, expected):
                raise TypeError(
                    f"{type(self).__name__}.action_space must be "
                    f"{expected.__name__} (or subclass), "
                    f"got {type(self.action_space).__name__}"
                )
        # extra_tool_schemas must not shadow model-facing top-level tool names:
        # they're appended alongside the action space's own tools, so a name
        # clash would be ambiguous at parse time. Action-batch child actions
        # are not top-level tools and must not globally reserve standalone
        # extras that happen to share a child-action name such as ``click``.
        if self.metadata.extra_tool_schemas:
            extra_names = {tool_schema_name(t) for t in self.metadata.extra_tool_schemas}
            top_level_tool_names = {
                tool_schema_name(schema) for schema in self.action_space.get_tool_schemas()
            }
            overlap = extra_names & top_level_tool_names
            if overlap:
                raise ValueError(
                    f"extra_tool_schemas names collide with standard action space: {overlap}"
                )

    def _cua_metadata(self) -> LiteCUAMetadata:
        if not isinstance(self.metadata, LiteCUAMetadata):
            raise TypeError(
                f"{type(self).__name__} requires LiteCUAMetadata for CUA tool "
                f"assembly; got {type(self.metadata).__name__}"
            )
        return self.metadata

    def _assemble_tool_schemas(self) -> list[dict[str, Any]]:
        """This adapter's ``metadata``, unwrapped into
        :func:`assemble_tool_schemas` — the shared schema-list behind every
        adapter's ``<tools>`` / ``# Tools`` block. Subclasses wrap the returned
        list in their own model-specific template string (essential per-family
        divergence).

        The pipeline itself is NOT a method: an API agent has no adapter and no
        ``self.metadata``, so the values travel as arguments. All this does is
        supply them from the adapter's own metadata.
        """
        metadata = self._cua_metadata()
        return assemble_tool_schemas(
            self.action_space,
            self.action_space.get_tool_schemas(),
            valid_actions=metadata.valid_actions,
            extra_tool_schemas=metadata.extra_tool_schemas,
        )

    def active_extra_tool_names(self) -> frozenset[str]:
        """Names of the standalone tools this sample actually advertises.

        Thin binding of :func:`lite.core.tools.extra_tools.active_extra_tool_names`
        to ``self.metadata``: the derivation itself is core-owned so the agent
        and env sides cannot drift. Families that expose native semantic
        spellings use this SAME set in their action-space schema filter, and it
        is forwarded to ``action_space.convert_tool_calls_from_agent`` so parser
        canonicalization can consult the same advertised surface where a family
        genuinely has a provider-spelling choice. Runtime env validation remains
        the authority for unsupported calls; this parser boundary only decides
        which provider spellings are canonical enough to become Lite tool calls.
        """
        return _active_extra_tool_names(self.metadata.extra_tool_schemas)

    def _admits_active_extra_tool_call(
        self,
        call: dict[str, Any],
        *,
        allowed_names: set[str] | frozenset[str] | None = None,
    ) -> bool:
        """Render-direction predicate: is this stored Lite call a standalone extra?

        Asked on the Lite -> agent side about a call the trajectory already
        holds as canonical, to choose between passing it through verbatim and
        projecting it through the action space. Nothing downstream can ask the
        env about it, so the answer is full schema satisfaction, delegated to
        the core admission owner ``extra_tool_name_and_arguments_are_admitted``.
        A "no" is not a drop: the caller falls back to
        ``action_space.convert_tool_calls_to_agent``.

        The parse direction asks a different question — see
        :meth:`_matches_active_extra_tool_schema_keys`.

        The call is read through the canonical accessors unguarded: a stored
        call that is not canonical is a producer bug, and both answers here lead
        straight back into ``tool_call_name`` — in the caller for a "yes", in
        ``convert_tool_calls_to_agent`` for a "no" — so swallowing the shape
        error would only move the same failure one frame later.
        """
        name = tool_call_name(call)
        arguments = tool_call_arguments(call)
        active_names: set[str] | frozenset[str] = self.active_extra_tool_names()
        if allowed_names is not None:
            active_names = set(active_names) & set(allowed_names)
        return isinstance(arguments, dict) and extra_tool_name_and_arguments_are_admitted(
            name,
            arguments,
            active_extra_tool_names=active_names,
            active_extra_tool_schemas=self.metadata.extra_tool_schemas,
        )

    def _matches_active_extra_tool_schema_keys(
        self,
        call: dict[str, Any],
        *,
        allowed_names: set[str] | frozenset[str] | None = None,
    ) -> bool:
        """Parse-direction predicate: does this model-emitted call target an env tool?

        Asked on the agent -> Lite side about untrusted model output, to send a
        name collision either to the env or to GUI conversion. Key shape is the
        whole test on purpose: full schema satisfaction here would route a
        type-wrong env call into GUI conversion, where the model never learns
        what was wrong. ``prepare_env_tool_calls()`` answers bad types and enums
        with env-owned feedback keyed to the call id instead.

        The render direction asks a different question — see
        :meth:`_admits_active_extra_tool_call`.

        ``call`` is agent-layer, so it is the bare ``{name, arguments}``
        projection every parser emits — never the canonical nested envelope.
        A non-dict or otherwise non-bare call answers "no" and is left to
        ``action_space.convert_tool_calls_from_agent``, which owns the bare-call
        contract and names the violation.
        """
        if not isinstance(call, dict):
            return False
        name = call.get("name")
        arguments = call.get("arguments")
        active_names: set[str] | frozenset[str] = self.active_extra_tool_names()
        if allowed_names is not None:
            active_names = set(active_names) & set(allowed_names)
        if name not in active_names:
            return False
        if not isinstance(arguments, dict):
            return False
        return any(
            tool_name_and_arguments_match_schema_route_keys(
                name=name,
                arguments=arguments,
                schema=schema,
            )
            for schema in self.metadata.extra_tool_schemas
            if tool_schema_name(schema) == name
        )

    def _route_agent_tool_calls_to_lite(
        self,
        tool_calls: list[dict[str, Any]],
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """``action_space.convert_tool_calls_from_agent`` with env extras winning
        a NAME COLLISION against the canonical GUI actions.

        An env may advertise a standalone tool whose name equals an action-batch
        child action but whose schema is completely different, such as a
        ``click`` tool keyed by an element id rather than screen coordinates.
        The construction-time guard in ``__post_init__`` cannot reject those:
        it compares extras against top-level action-space tool names, not child
        action names. Without this routing, such a call is fed into GUI
        conversion and fails before env-owned feedback can report the schema
        problem.

        ``metadata.extra_tool_schemas`` is the single source of truth for the
        standalone surface. Top-level action-space schema names still win, so a
        family's own wrapper can never be shadowed. Action-batch child action
        collisions are resolved by key shape: a key-matching env extra is sent
        to the environment for final validation/feedback, while a coordinate
        shaped GUI call still goes through GUI conversion. Extras split GUI
        runs, matching the "standalone extras are never crossed by the merge"
        rule the action-batch mergers already document.
        """
        active_extras = self.active_extra_tool_names()
        kwargs.setdefault("active_extra_tool_names", active_extras)
        kwargs.setdefault(
            "active_extra_tool_schemas",
            list(self.metadata.extra_tool_schemas),
        )
        # TOOL layer, not ACTION layer: only a name the action space itself
        # EMITS as a top-level tool may shadow a standalone extra. Reading
        # ``get_declared_action_schema_names()`` here instead would exclude GUI
        # *child* action names too, silently routing an advertised env extra named
        # like a child action back into GUI conversion. For action-batch families the
        # two accessors are not interchangeable: one top-level tool can carry
        # many child actions.
        top_level_tool_names = {
            tool_schema_name(schema) for schema in self.action_space.get_tool_schemas()
        }
        extra_names = {
            tool_schema_name(schema)
            for schema in self.metadata.extra_tool_schemas
            if tool_schema_name(schema) in active_extras
            and tool_schema_name(schema) not in top_level_tool_names
        }

        out: list[dict[str, Any]] = []
        action_run: list[dict[str, Any]] = []

        def _flush() -> None:
            if action_run:
                out.extend(
                    self.action_space.convert_tool_calls_from_agent(
                        action_run,
                        **kwargs,
                    )
                )
                action_run.clear()

        for call in tool_calls:
            if self._matches_active_extra_tool_schema_keys(call, allowed_names=extra_names):
                _flush()
                # Matching the schema route keys already established that this
                # is a bare ``{name, arguments}`` dict whose name is an active
                # extra, so read it directly.
                name = call["name"]
                arguments = call["arguments"]
                call_id = call.get("call_id")
                out.append(
                    make_tool_call(
                        name,
                        arguments,
                        call_id=call_id if isinstance(call_id, str) else None,
                    )
                )
            else:
                action_run.append(call)
        _flush()
        return out

    # -------------------------------------------------------------------------
    # Sample-level contract
    # -------------------------------------------------------------------------

    def select_action_batch_image_indices(
        self,
        *,
        tool_call: dict[str, Any],
        tool_result: LiteToolResult,
        result_image_indices: tuple[int, ...],
    ) -> tuple[int, ...]:
        """Model-visible image policy for one action-batch tool result.

        An action batch runs N actions and the env may return one frame per
        action. The trajectory stores EVERY returned frame regardless; this
        hook only chooses which of them the next model turn sees. Frames it
        drops keep their trajectory indices — the referenced indices jump over
        them and their ``processed_images`` slots stay ``None`` — so overriding
        this never loses data and never renumbers anything.

        Default: the FINAL frame, i.e. the state after the whole batch ran.

        ``tool_call`` and ``tool_result`` are passed so an override can decide
        from what the batch actually did (its child actions) and what the env
        answered (text, error), not just from the frame count. Non-batch
        results never reach here; they show all their frames.
        """
        del tool_call
        del tool_result
        return result_image_indices[-1:]

    def process_image(self, img: AdapterImageCarrier) -> AdapterImageCarrier:
        """Per-image transformation pipeline.

        Two-stage pipeline:
        - **Stage 1 — target-resize**: exact stretch to ``self.resolution``
          (when set). This is the canonical model-operates-in-this-frame
          step; the coord space the model emits and the env-side action
          denormalization both reference this resolution. The caller picks
          a target whose AR matches the env to avoid distortion.
        - **Stage 2 — adapter-specific transform**: subclasses override
          :meth:`_process_image_after_target` to satisfy their own vision
          processor's geometry constraints.

        ``process_image`` itself is the base-owned template method used by
        unroll and rollout callers. Subclasses should override
        :meth:`_process_image_after_target` unless they need to replace the
        whole pipeline.

        Called for every model-visible image carrier. PIL inputs are transformed
        and stored in ``AgentSample.processed_images`` at the same index as the
        raw ``lite_sample.images`` entry. Path-string carriers are an explicit
        external image-store boundary: adapters cannot resize them here, so they
        pass through unchanged. Stored-but-invisible frames may keep ``None``
        placeholders.

        Must be deterministic and side-effect-free: the same input PIL
        produces the same output PIL byte-for-byte across calls. The
        radix segmenter relies on this so that an image processed once
        per trajectory has stable indices across all step views.
        """
        if isinstance(img, str):
            return img
        if self.resolution is not None:
            img = _resize_to_target(img, self.resolution)
        return self._process_image_after_target(img)

    def _process_image_after_target(self, img: Image.Image) -> Image.Image:
        """Stage-2 hook: adapter-specific transform after target-resize.

        Default identity. Subclasses override this to apply their own
        vision-processor-specific resize. Receives an image already
        downscaled to ``self.resolution`` (if set).
        """
        return img

    @abstractmethod
    def render_step(
        self,
        sample: LiteSample,
        k: int,
        processed: list[AdapterImageCarrier | None],
    ) -> AgentStep:
        """Hook 2 — render the messages the model would see at turn ``k``.

        Subclasses implement this. Responsibilities:
        - Apply the adapter's :attr:`protocol` to truncate / window /
          summarize history up to and including turn ``k``.
        - Walk each in-window message through
          :meth:`_convert_message_to_agent` (or a custom per-message
          helper) to produce the agent-side rendering.
        - Prepend system prompt(s) per the adapter's convention.
        - Preserve native ``ImageContent = {"type": "image", "index": int}``
          parts so consumers can recover image refs by walking messages.

        Args:
            sample: The full :class:`LiteSample` trajectory.
            k: 1-indexed turn number to render. ``k == count_sample_turns(sample)``
               renders the final turn (post-rollout, full trajectory) or
               the partial turn at predict time (sample ends in user msg).
            processed: Index-aligned with ``sample.images``. Model-visible
               entries are processed PIL images or explicit external image-store
               carriers; unreferenced raw frames may be ``None``.

        Returns:
            :data:`AgentStep` — list of rendered :data:`AgentMessage`
            dicts, pre-``apply_chat_template``.
        """
        raise NotImplementedError

    def unroll(self, sample: LiteSample) -> AgentSample:
        """Skeleton — generic; do NOT override.

        Composes :meth:`process_image` and :meth:`render_step` to produce
        a trajectory-level :class:`AgentSample`. Each turn ``k`` from 1
        to ``count_sample_turns(sample)`` becomes an entry in
        ``AgentSample.steps``. Image carriers live once on
        ``AgentSample.processed_images``.
        """
        referenced = set(_image_refs.referenced_image_indices_in_message_order(sample.messages))
        processed: list[AdapterImageCarrier | None] = [
            self.process_image(img) if i in referenced else None
            for i, img in enumerate(sample.images)
        ]
        n = count_sample_turns(sample)
        steps: list[AgentStep] = []
        for k in range(1, n + 1):
            step = self.render_step(sample, k, processed)
            for idx in _image_refs.referenced_image_indices_in_message_order(step):
                if idx >= len(processed):
                    raise ValueError(
                        f"{type(self).__name__}.render_step(k={k}) emitted image index "
                        f"{idx}, but processed_images has length {len(processed)}"
                    )
                if processed[idx] is None:
                    raise ValueError(
                        f"{type(self).__name__}.render_step(k={k}) emitted image index "
                        f"{idx}, but processed_images[{idx}] was not prepared from "
                        "the source sample"
                    )
            steps.append(step)
        meta = sample.metadata.to_dict() if sample.metadata is not None else {}
        return AgentSample(
            processed_images=processed,
            steps=steps,
            metadata=meta,
        )

    # -------------------------------------------------------------------------
    # Per-message helpers (kept; called from render_step + at boundaries)
    # -------------------------------------------------------------------------

    def convert_message_to_agent(
        self,
        message: LiteMessage,
        **kwargs,
    ) -> AgentMessage:
        """Convert a CUA-lite message to agent format.

        Assistant messages with a matching ``raw_response`` sidecar use the
        adapter registry's public key matcher to short-circuit to
        ``{role, content=[text: raw]}``. Persisted canonical no-tool finals are
        rendered canonically, ignoring any sidecar on that shape.
        Mismatch → deduped warn + canonical fallback via
        :meth:`_convert_message_to_agent`. Subclasses implement the canonical
        hook; override this public method to opt out of the short-circuit
        entirely (e.g. :class:`AsIsAdapter`).
        """
        is_canonical_final = is_canonical_no_tool_call_final_message(message)
        if message.get("role") == ASSISTANT_ROLE:
            self._require_standalone_tool_schemas(message.get("tool_calls") or [])
            if is_canonical_final and message.get("raw_response") is not None:
                message = {**message}
                message.pop("raw_response", None)

        raw = message.get("raw_response")
        if raw is not None and message.get("role") == ASSISTANT_ROLE and not is_canonical_final:
            expected = self._registry_key
            actual = raw.get("adapter_key")
            if AgentAdapterRegistry.raw_response_key_matches(actual, expected):
                replay: AgentMessage = {
                    "role": ASSISTANT_ROLE,
                    "content": [{"type": TEXT_PART, "text": raw["text"]}],
                }
                # A sidecar whose text embeds the whole reply carries no
                # ``reasoning_content`` and needs none -- the raw string already
                # holds the ``<think>`` tokens. A family that generated against an
                # OPEN block trims that half out of the text and records it here
                # instead, so the chat template can rebuild the block.
                if reasoning := raw.get("reasoning_content"):
                    replay["reasoning_content"] = reasoning
                return replay
            _warn_adapter_key_mismatch(expected, actual)
        return self._convert_message_to_agent(message, **kwargs)

    def _require_standalone_tool_schemas(self, tool_calls: list[dict[str, Any]]) -> None:
        """Validate render-only structural invariants for action-batch calls.

        Availability belongs to env/data validation. Rendering may replay a
        canonical tool call even when the current metadata did not advertise it;
        schema rendering remains controlled by ``metadata.extra_tool_schemas``.
        Action-batch child availability failures are preserved as env feedback, not
        rejected during later re-rendering.
        """
        for call in tool_calls:
            if tool_call_name(call) in LITE_ACTION_BATCH_TOOL_NAMES:
                feedback = self._action_batch_feedback(call)
                if feedback is not None:
                    _warn_invalid_action_batch_child(type(self).__name__, feedback)

    def _action_batch_feedback(self, tool_call: dict[str, Any]) -> str | None:
        """Return model-readable feedback for one canonical action-batch, or None.

        Wording matches the env side (``action_batch_structure_error_message``
        and ``nested_extra_tool_action_batch_child_message``) so one misplaced call
        reads the same whether the model meets it as a step result or in its own
        rendered history.

        The two failures are named apart because they are different mistakes. A
        hallucinated child is a name that exists nowhere; an ACTIVE extra tool is
        a real, offered TOOL that the model put one layer too deep. It is named
        as such rather than re-homed to the top level: ``extra_tools`` gates
        tools and ``valid_actions`` gates actions, so silently moving a call
        across that line would hide the layer confusion instead of making it
        recoverable — and the canonical action-batch is contractually preserved
        intact for the env to answer.

        Raises:
            ValueError: the action-batch payload has no renderable shape at all
                (``arguments``/``actions`` malformed, or an unnamed child).
                Nothing downstream can render that, so it stays fatal.
        """
        name = tool_call_name(tool_call)
        children, error = validate_lite_action_batch_structure(
            name,
            tool_call_arguments(tool_call),
        )
        if error is not None:
            raise ValueError(
                f"{type(self).__name__} canonical {name!r} action-batch is not "
                f"renderable: {error.reason}"
            )
        # A child naming an action the batch tool does not carry is no longer an
        # envelope error -- env ingress forwards it so the env can answer it per
        # action. The name check is its own function now, and this renderer is
        # its second consumer.
        name_errors = lite_action_batch_child_name_errors(name, children)
        if not name_errors:
            return None
        error = name_errors[min(name_errors)]
        child = error.child_action_name
        if child is None:
            return None
        if child in self.active_extra_tool_names():
            return (
                f"invalid action: {child}; {name}.actions cannot contain "
                f"standalone extra tool {child}"
            )
        return f"invalid action: {child}; {error.reason}"

    @abstractmethod
    def _convert_message_to_agent(
        self,
        message: LiteMessage,
        **kwargs,
    ) -> AgentMessage:
        """Canonical structural CUA-lite → agent conversion (subclass hook)."""
        raise NotImplementedError

    @abstractmethod
    def convert_message_from_agent(
        self,
        message: AgentMessage,
        **kwargs,
    ) -> LiteMessage:
        """
        Convert a message from agent format to CUA-lite format.
        """
        raise NotImplementedError

    @abstractmethod
    def parse_raw_assistant_response(
        self,
        response: str,
        **kwargs,
    ) -> AgentMessage:
        """
        Parse a raw assistant response to an AgentMessage.

        The returned dict follows the LiteAssistantMessage contract:

          * ``content`` is a list of structured parts — ``TextContent``
            for plain text (QA answers, generic output, and no-tool final
            prose), ``ActionDescriptionContent`` only for action-turn
            narration paired with ``tool_calls``,
            ``InlineReasoningContent`` for prompted CoT,
            ``HistorySummaryContent`` for trajectory rolling-summary.
          * ``tool_calls`` (optional): structured action calls.
          * ``reasoning_content`` (optional): top-level **native** reasoning
            channel — only set by adapters whose model family exposes one.
            Prompted / inline CoT goes into ``InlineReasoningContent`` inside
            ``content`` instead.
        """
        raise NotImplementedError


# =============================================================================
# Built-in Adapter: AsIsAdapter (Pass-through for understanding/grounding)
# =============================================================================


@dataclasses.dataclass
class AsIsAdapter(BaseAgentAdapter, key="as_is"):
    """
    Pass-through adapter that returns data unchanged.

    Use for:
    - understanding tasks (no tools)
    - grounding/bbox tasks (CUA-lite format is already correct)
    - grounding/point tasks (CUA-lite format is already correct)

    Single-turn (T=1) by construction — :meth:`render_step` is called
    exactly once and returns the entire messages list verbatim.
    """

    action_space: BaseActionSpace = dataclasses.field(default_factory=_default_lite_action_space)
    protocol: BaseProtocol = dataclasses.field(default_factory=_default_full_history_protocol)

    def render_step(
        self,
        sample: LiteSample,
        k: int,
        processed: list[AdapterImageCarrier | None],
    ) -> AgentStep:
        """T=1 pass-through. The whole conversation is one turn."""
        if k != 1:
            raise ValueError(
                f"AsIsAdapter is single-turn (T=1); got k={k}. "
                f"Caller likely miscounted turns or used the wrong adapter."
            )
        return [copy.deepcopy(m) for m in sample.messages]

    def convert_message_to_agent(
        self,
        message: LiteMessage,
        **kwargs,
    ) -> LiteMessage:
        """Pass-through: bypass base short-circuit to preserve full structure
        (``tool_calls`` / ``reasoning_content`` / all content parts)."""
        if message.get("role") == ASSISTANT_ROLE:
            self._require_standalone_tool_schemas(message.get("tool_calls") or [])
        return self._convert_message_to_agent(message, **kwargs)

    def _convert_message_to_agent(
        self,
        message: LiteMessage,
        **kwargs,
    ) -> LiteMessage:
        """Return message unchanged (deep copy)."""
        return copy.deepcopy(message)

    def convert_message_from_agent(
        self,
        message: AgentMessage,
        **kwargs,
    ) -> LiteMessage:
        """Return message unchanged (deep copy)."""
        return copy.deepcopy(message)

    def parse_raw_assistant_response(
        self,
        response: str,
        **kwargs,
    ) -> dict[str, Any]:
        """Return minimal assistant message with raw text."""
        return {
            "role": ASSISTANT_ROLE,
            "content": [{"type": TEXT_PART, "text": response}],
        }
