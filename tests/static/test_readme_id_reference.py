"""The root README's id reference must match the live registries.

README.md's Quick Start carries a collapsed list of every ``--model-id`` and
``--env-id``. A hand-written list beside a registry drifts -- this repo has
already shipped a README whose agent-family section predated a live factory id
-- so this pins both.

Two shapes need undoing before comparison, because both exist to keep the block
readable rather than to describe the wire:

* Model ids are brace-expanded (``Qwen/Qwen3-VL-{2B,4B}-Instruct``), and are
  matched EXACTLY -- ``AGENTS`` is a closed hand-maintained list, so an
  unlisted id is a real omission. The summary line carries no counts: the
  family names a reader would search for are already in the "Any CUA" bullet
  and the two "Supported" sections, which are not collapsed.
* Env ids are matched by PREFIX. ``lite.cuaworld`` alone expands to 39
  per-application ids, and ``registered_env_ids()`` returns whichever leaves the
  importing process happens to have registered. Listing leaves would put 39
  application names in a README and would break on every new app; listing the
  family and asserting coverage is the invariant that actually holds.
"""

from __future__ import annotations

import itertools
import re

import lite.gym as gym
from lite.agents.factory import AGENTS
from lite.utils.path import project_root

_README = project_root() / "README.md"
_DETAILS = re.compile(
    r"<details>\s*\n<summary><b>All agent ids and env ids</b></summary>.*?</details>",
    re.S,
)
_CODE = re.compile(r"`([^`]+)`")
_TEST_ENV_PREFIXES = ("_", "faketest_")
_ENV_APP_SUFFIX = ".<app>"
_NAMESPACE_ENVS_COVERED_BY_CHILDREN = {"cua.bench"}


def _reference_block() -> str:
    match = _DETAILS.search(_README.read_text())
    assert match, "README.md lost the collapsed id reference in Quick Start"
    return match.group(0)


def _expand(token: str) -> list[str]:
    """``a-{x,y}-b`` -> ``[a-x-b, a-y-b]``; nested groups expand as a product."""
    groups = re.findall(r"\{([^}]*)\}", token)
    if not groups:
        return [token]
    template = re.sub(r"\{[^}]*\}", "{}", token)
    return [template.format(*combo) for combo in itertools.product(*(g.split(",") for g in groups))]


def _listed_tokens() -> set[str]:
    block = _reference_block()
    return {expanded for token in _CODE.findall(block) for expanded in _expand(token)}


def _env_token_family(token: str) -> str:
    """README ``foo.<app>`` shorthand names the ``foo`` env family."""
    if token.endswith(_ENV_APP_SUFFIX):
        return token.removesuffix(_ENV_APP_SUFFIX)
    return token


def test_readme_lists_every_model_id() -> None:
    missing = sorted(set(AGENTS) - _listed_tokens())
    assert not missing, (
        "model ids registered in lite.agents.factory.AGENTS but absent from the "
        "README id reference:\n  " + "\n  ".join(missing)
    )


def test_readme_lists_no_retired_model_id() -> None:
    """The other direction: a listed id the factory no longer offers."""
    # A model id is an HF repo id (``org/name``) or an API family id. Repo
    # paths in the same block also contain "/", so exclude those explicitly.
    listed_models = {
        token
        for token in _listed_tokens()
        if (
            ("/" in token and not token.startswith(("lite/", "scripts/", "docs/")))
            or token.startswith(("gpt-", "claude-", "gemini-"))
        )
    }
    stale = sorted(listed_models - set(AGENTS))
    assert not stale, (
        "README id reference lists model ids that are no longer registered:\n  "
        + "\n  ".join(stale)
    )


def test_every_env_id_is_covered_by_a_listed_family() -> None:
    """Every registered env id must fall under some listed token.

    Prefix, not equality: ``lite.cuaworld.<app>`` and the ``browsergym.*`` /
    ``cua.bench.local.*`` leaves are covered by their family entry.
    """
    listed = _listed_tokens()

    def covered(env_id: str) -> bool:
        for token in listed:
            family = _env_token_family(token)
            if env_id == family or env_id.startswith(family + "."):
                return True
            if env_id in _NAMESPACE_ENVS_COVERED_BY_CHILDREN and token.startswith(env_id + "."):
                return True
        return False

    readme_env_ids = [
        env_id
        for env_id in gym.registry.registered_env_ids()
        if not env_id.startswith(_TEST_ENV_PREFIXES)
    ]
    uncovered = sorted(e for e in readme_env_ids if not covered(e))
    assert not uncovered, (
        "env ids registered in lite.gym that no README entry covers:\n  " + "\n  ".join(uncovered)
    )
