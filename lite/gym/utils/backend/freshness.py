"""Docker image freshness: refuse to launch a container whose image is stale.

A container's image can silently fall behind its Dockerfile/patches after an edit
you forgot to rebuild. ``ContainerImage.ensure_runnable()`` (called from the launch
helpers in ``lite.gym.utils.backend.docker``) recomputes a hash of the build SOURCES and
compares it to the ``lite.src_hash`` label the image's install.sh stamped via the
same hash, so build and launch can't drift. See /docs/envs.md#image-build-and-freshness.

    python -m lite.gym.utils.backend.freshness hash <env_id>   # used by install.sh
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import stat
import subprocess
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from pathlib import Path

from lite.gym.errors import CapacityExhausted, EnvDepsMissingError
from lite.utils.path import project_root

# Private on purpose: this constant has exactly one reader (``ensure_runnable``
# below) and zero Python importers, so it is NOT the label's source of truth.
# ``lite/gym/scripts/image_build.sh`` IS — it hardcodes the literal
# ``lite.src_hash`` ~14 times as BOTH the writer (``--label`` stamping) and a
# reader (``docker image inspect --format '{{index .Config.Labels ...}}'`` in
# the build / push / pull gates), and ``lite/gym/envs/lite/cuaworld/scripts/
# install.sh`` parses it once more (``${img_label#lite.src_hash=}``). Renaming
# the label means editing all of those together, in two languages — and the
# failure is SILENT in the safe direction only by luck: a name mismatch reads as
# "no label" → STALE → forced rebuilds, never a stale image accepted as fresh.
# Do not "fix" the duplication by exporting this constant; bash cannot import it.
_LABEL_KEY = "lite.src_hash"
_INSPECT_TIMEOUT_S = 20.0

# Repo root by marker lookup (pyproject.toml + lite/), not __file__-depth
# counting — depth counting silently breaks the moment this module moves.
_REPO_ROOT = project_root()

# `python -m lite.gym.utils.backend.freshness ...` executes this file as
# `__main__`; env-local providers import the canonical module path. Alias the
# running module before provider import so `ContainerImage` has one class
# identity and the provider contract can stay strict.
if __name__ == "__main__":
    sys.modules["lite.gym.utils.backend.freshness"] = sys.modules[__name__]

# Build junk that never affects the image (a mini, common-case .dockerignore).
_IGNORE_SEGMENTS = frozenset({"__pycache__", ".pytest_cache", ".git", ".mypy_cache"})


def _ignored(rel_posix: str) -> bool:
    # `.gitignore` is a git artifact, never COPY'd into any image.
    return (
        rel_posix.endswith((".pyc", ".pyo"))
        or rel_posix.rsplit("/", 1)[-1] == ".gitignore"
        or bool(_IGNORE_SEGMENTS.intersection(rel_posix.split("/")))
    )


def _is_dockerfile(rel_posix: str) -> bool:
    name = rel_posix.rsplit("/", 1)[-1]
    return name == "Dockerfile" or name.startswith("Dockerfile.")


_HEREDOC_RE = re.compile(
    r"<<-?\s*(?P<quote>['\"]?)(?P<word>[^\s'\"<>|&;()]+)(?P=quote)"
)


def _dockerfile_heredoc_delimiters(line: str) -> tuple[str, ...]:
    return tuple(match.group("word") for match in _HEREDOC_RE.finditer(line))


def _normalize_dockerfile_bytes(raw: bytes) -> bytes:
    """Hash a Dockerfile by SEMANTIC content, not raw bytes: drop standalone
    Dockerfile-level ``#`` comment lines + blank lines (a comment/whitespace-only
    edit produces a byte-identical image, so it must not force a rebuild). PRESERVE:
    (1) parser directives (``# syntax=``/``# escape=``/``# check=``) that appear
    before the first instruction — they DO affect the build; (2) anything inside a
    line-continuation (``\\``), because a ``#`` there is a SHELL comment that is part
    of the built RUN command; (3) anything inside a BuildKit heredoc, because those
    lines become command/script/file content. Conservative: can only ever remove
    pure Dockerfile comments/blanks, never an instruction or heredoc content — so it
    never UNDER-triggers a rebuild."""
    out: list[str] = []
    in_continuation = False
    heredoc_delimiters: list[str] = []
    seen_instruction = False
    for line in raw.decode("utf-8", "surrogateescape").splitlines():
        if heredoc_delimiters:
            out.append(line)
            if line.strip() == heredoc_delimiters[0]:
                heredoc_delimiters.pop(0)
            continue
        if not in_continuation:
            stripped = line.strip()
            if stripped == "":
                continue  # drop instruction-level blank line
            if stripped.startswith("#"):
                directive = stripped[1:].lstrip().split("=", 1)[0].strip().lower()
                if not seen_instruction and directive in ("syntax", "escape", "check"):
                    out.append(line)  # parser directive affects the build — keep
                continue  # drop the Dockerfile-level comment
            seen_instruction = True
        out.append(line)
        if not in_continuation:
            heredoc_delimiters.extend(_dockerfile_heredoc_delimiters(line))
        in_continuation = line.rstrip().endswith("\\")
    return ("\n".join(out) + "\n").encode("utf-8", "surrogateescape")


def _excluded(rel_posix: str, basename: str, exclude: tuple[str, ...]) -> bool:
    for item in exclude:
        if "/" in item:
            if rel_posix == item or rel_posix.endswith(f"/{item}"):
                return True
        elif basename == item:
            return True
    return False


TreeSkip = Callable[[Path, Path], bool]


def tree_content_hash(
    root: Path,
    *,
    roots: tuple[str, ...] = (".",),
    skip: TreeSkip | None = None,
    missing: str = "error",
) -> str:
    """Content digest for non-repo build inputs.

    Env image specs use this for local checkouts or material trees that affect an
    image but cannot be represented as repo-relative ``sources``. The digest
    includes relative paths, file modes, symlink targets, and bytes. ``skip`` is
    env-owned policy: keep it local to the spec that owns the external tree.

    ``missing``:
      - ``"error"``: missing roots are hard errors;
      - ``"marker"``: missing roots contribute an absence marker.
    """
    if missing not in {"error", "marker"}:
        raise ValueError(f"unknown tree_content_hash missing policy: {missing!r}")
    root = Path(root).expanduser().resolve()
    h = hashlib.sha256()

    def add(path: Path) -> None:
        if skip and skip(root, path):
            return
        st = path.lstat()
        rel = path.relative_to(root).as_posix()
        mode = stat.S_IMODE(st.st_mode)
        if stat.S_ISDIR(st.st_mode):
            h.update(f"D {rel} {mode:o}\0".encode())
            for child in sorted(path.iterdir(), key=lambda p: p.name):
                add(child)
        elif stat.S_ISLNK(st.st_mode):
            h.update(f"L {rel} {mode:o}\0".encode())
            h.update(os.readlink(path).encode())
            h.update(b"\0")
        elif stat.S_ISREG(st.st_mode):
            h.update(f"F {rel} {mode:o}\0".encode())
            h.update(path.read_bytes())
            h.update(b"\0")
        else:
            kind = stat.S_IFMT(st.st_mode)
            h.update(f"O {rel} {kind:o} {mode:o}\0".encode())

    for rel in roots:
        path = root / rel
        if path.exists() or path.is_symlink():
            add(path)
        elif missing == "marker":
            h.update(f"A {rel}\0".encode())
        else:
            raise FileNotFoundError(f"tree_content_hash root is missing: {rel}")
    return h.hexdigest()


def _source_files(
    rel_sources: tuple[str, ...], exclude: tuple[str, ...]
) -> list[tuple[str, Path]]:
    """Sorted ``(repo-relative posix path, absolute Path)`` for every source file,
    skipping build junk and ``exclude``d files.

    ``exclude`` entries without ``/`` match basenames for historical specs such as
    ``server.py``. Entries containing ``/`` match a repo-relative path or suffix;
    prefer that form for new specs when only one copied file should be excluded.

    A declared source that does not exist is a hard error, NOT a silent skip:
    skipping would drop a real build input from the hash, so a rename/typo would
    make every already-built image keep reading FRESH forever — the one failure
    direction this module exists to prevent.
    """
    files: list[Path] = []
    for rel in rel_sources:
        p = _REPO_ROOT / rel
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            files.extend(f for f in p.rglob("*") if f.is_file())
        else:
            raise FileNotFoundError(
                f"image_freshness source {rel!r} does not exist. "
                "Fix the image spec — a missing source would "
                "silently stop contributing to the hash."
            )
    return sorted(
        (f.relative_to(_REPO_ROOT).as_posix(), f)
        for f in files
        if not _ignored(f.relative_to(_REPO_ROOT).as_posix())
        and not _excluded(f.relative_to(_REPO_ROOT).as_posix(), f.name, exclude)
    )


def _hash_fingerprint(fingerprint: tuple[tuple[str, int, int, str], ...]) -> str:
    """sha256 over every file (repo-relative path + bytes, sorted).

    This deliberately does not cache by mtime/size. Several generated lock files
    can be rewritten with identical byte length inside one timestamp tick; a
    cached fingerprint would then keep returning the old digest and mark stale
    assets/images as fresh.
    """
    h = hashlib.sha256()
    for rel_posix, _mtime_ns, _size, abs_path in fingerprint:
        h.update(rel_posix.encode())
        h.update(b"\0")
        raw = Path(abs_path).read_bytes()
        # Dockerfiles are hashed by semantic content so comment/blank-only edits
        # (which yield a byte-identical image) don't spuriously force a rebuild.
        if _is_dockerfile(rel_posix):
            raw = _normalize_dockerfile_bytes(raw)
        h.update(raw)
        h.update(b"\0")
    return h.hexdigest()


def _hash_sources(rel_sources: tuple[str, ...], exclude: tuple[str, ...]) -> str:
    fingerprint = tuple(
        (rel_posix, st.st_mtime_ns, st.st_size, str(f))
        for rel_posix, f in _source_files(rel_sources, exclude)
        for st in (f.stat(),)
    )
    return _hash_fingerprint(fingerprint)


class _DockerCliMissing(RuntimeError):
    pass


class _ImageInspectTransientError(RuntimeError):
    pass


class _ImageInspectTerminalError(RuntimeError):
    pass


def _process_output(exc: subprocess.CalledProcessError) -> str:
    chunks = [exc.stderr, exc.stdout, exc.output]
    return "\n".join(str(c) for c in chunks if c).strip()


def _image_missing_from_inspect_error(message: str) -> bool:
    lower = message.lower()
    return (
        "no such image" in lower
        or "no such object" in lower
        or "reference does not exist" in lower
    )


def _image_inspect_error_is_transient(message: str) -> bool:
    lower = message.lower()
    return any(
        needle in lower
        for needle in (
            "cannot connect to the docker daemon",
            "error during connect",
            "connection refused",
            "context deadline exceeded",
            "i/o timeout",
            "temporarily unavailable",
            "temporary docker inspect failure",
        )
    )


def _image_label(image: str, key: str) -> str | None:
    """The image's value for label ``key``, or ``None`` if the image is absent or
    unlabeled (both mean "rebuild needed").

    Unexpected ``docker image inspect`` failures are not collapsed into
    ``None``: callers need to distinguish a true missing image from a transient
    daemon/CLI failure.
    """
    try:
        r = subprocess.run(
            ["docker", "image", "inspect", "--format",
             f"{{{{index .Config.Labels \"{key}\"}}}}", image],
            capture_output=True, text=True, check=True,
            timeout=_INSPECT_TIMEOUT_S,
        )
    except FileNotFoundError as exc:
        raise _DockerCliMissing("docker CLI not found") from exc
    except subprocess.TimeoutExpired as exc:
        raise _ImageInspectTransientError(
            f"docker image inspect timed out after {_INSPECT_TIMEOUT_S:.0f}s"
        ) from exc
    except subprocess.CalledProcessError as exc:
        message = _process_output(exc)
        if _image_missing_from_inspect_error(message):
            return None
        error_cls = (
            _ImageInspectTransientError
            if _image_inspect_error_is_transient(message)
            else _ImageInspectTerminalError
        )
        raise error_cls(
            message or f"docker image inspect exited with {exc.returncode}"
        ) from exc
    out = r.stdout.strip()
    return out if out and out != "<no value>" else None


@dataclass(frozen=True)
class ContainerImage:
    """A docker image bound to the build sources that produce it. Freshness is a
    method here, so a launch helper holding one can't forget to check it."""

    tag: str
    sources: tuple[str, ...]        # repo-relative build-source paths to hash
    install: str                    # install.sh hint shown on failure
    see: str                        # repo-relative README path shown on failure
    exclude: tuple[str, ...] = ()   # basenames under ``sources`` NOT to hash —
                                    # build-time staged copies, files bind-mounted
                                    # over the baked copy at run time, or host-only
                                    # members of a directory source (see THE RULE
                                    # below); entries containing / match
                                    # repo-relative path suffixes.
    extra_hash_inputs: tuple[str, ...] = ()  # non-git build inputs, e.g. build args
                                             # or pinned out-of-tree baked assets.

    def __post_init__(self) -> None:
        for value in self.extra_hash_inputs:
            if not isinstance(value, str):
                raise TypeError(
                    "ContainerImage.extra_hash_inputs must contain only strings"
                )

    def src_hash(self) -> str:
        source_hash = _hash_sources(self.sources, self.exclude)
        if not self.extra_hash_inputs:
            return source_hash
        return hashlib.sha256(
            json.dumps(
                [source_hash, *self.extra_hash_inputs],
                separators=(",", ":"),
            ).encode()
        ).hexdigest()

    def ensure_runnable(self) -> None:
        """Raise ``EnvDepsMissingError`` unless the image exists AND its
        ``lite.src_hash`` label matches the current build sources.

        A transient inspect failure raises ``CapacityExhausted`` instead of
        pretending the image is missing/stale.
        """
        want = self.src_hash()
        try:
            have = _image_label(self.tag, _LABEL_KEY)
        except _DockerCliMissing as exc:
            raise EnvDepsMissingError(
                what=f"Docker CLI is not available while checking image {self.tag!r}",
                install=self.install,
                see=self.see,
            ) from exc
        except _ImageInspectTransientError as exc:
            raise CapacityExhausted(
                what=f"docker image inspect failed for {self.tag!r}: {exc}",
                retry_after_s=2.0,
            ) from exc
        except _ImageInspectTerminalError as exc:
            raise EnvDepsMissingError(
                what=f"Docker image {self.tag!r} cannot be inspected: {exc}",
                install=self.install,
                see=self.see,
            ) from exc
        if have == want:
            return
        detail = (
            "image missing, or built before freshness tracking (no label)"
            if have is None
            else "sources changed since the image was built — the hash is over the "
                 "RAW BYTES of every source and only Dockerfiles drop comments, so a "
                 "comment-only edit to a baked .py/.sh is enough to land here"
        )
        raise EnvDepsMissingError(
            what=f"Docker image {self.tag!r} is STALE — {detail}. Rebuild first.",
            install=self.install,
            see=self.see,
        )


# ─── THE RULE for what belongs in a spec's ``sources`` ───────────────────────
# Hash a file IFF its CONTENT can change what ends up INSIDE the image.
#
# The two failure directions are NOT symmetric, so the rule is applied
# asymmetrically when a file is arguable:
#   * over-hashing (a file that cannot change the image)  → a spurious rebuild.
#     Annoying, never wrong.
#   * under-hashing (a file that CAN change the image)    → a STALE image reports
#     FRESH and silently runs forever. This is the failure this module exists to
#     prevent. When in doubt, INCLUDE.
#
# ``scripts/install.sh`` — READ THIS BEFORE "OPTIMIZING" IT OUT.
# An install.sh is a hash input for exactly one reason: some of them ASSEMBLE THE
# BUILD CONTEXT (stage/download/generate the files docker then COPYs) or pass
# build-args, so their content literally decides image contents.
#
# It is TEMPTING to drop install.sh from a hash because most of an install.sh is
# host-side orchestration (pip-installing host verifier deps into the repo venv,
# mode dispatch, pull/push, freshness probes), and editing that re-stales every
# image the script builds — 40 of them for cuaworld — for zero image change. That
# cost is real, but dropping the file is the UNDER-hashing direction: it would
# also stop catching an edit to the staging list a few lines away, i.e. exactly a
# silent-stale. The correct way to make an env quiet is to move the host-only code
# OUT of install.sh into a sibling helper that is not a hash input — never to stop
# hashing a script that stages the build context.
#
# Do NOT add host-only files (host verifier deps, task catalogs, evaluators,
# configs read at run time) as sources — none of them can change image contents,
# and each one buys a permanent stream of spurious full rebuilds. Importers are
# host-only unless they generate baked build-context inputs; CUAGym is the
# exception: CUA-Gym's importer produces apps.txt, which decides which mocks are
# staged into the image, so its env-local spec hashes that importer surface.
#
# When a source is a DIRECTORY, the same rule applies member by member, and the
# lever is ``exclude`` — NOT dropping the directory. Keeping the directory listed
# means a newly added file is hashed by default (over-hash, safe); naming its
# individual members instead would make a new stager silently unhashed. CUAGym's
# spec does this: ``scripts/utils`` stays a source, with its two host-only
# members excluded and the reason recorded per file.

ImageProvider = Callable[[str], ContainerImage]


class UnknownImageFreshnessProvider(LookupError):
    """No env-local freshness provider exists for the requested env id."""

_ENVS_ROOT = _REPO_ROOT / "lite/gym/envs"
_PROVIDER_ATTR = "image_for"
_SKIP_PROVIDER_SEGMENTS = frozenset({".cache", "_vendor", "__pycache__"})


def _provider_module_path(env_parts: tuple[str, ...], *, src: bool) -> str:
    suffix = ".src.image_spec" if src else ".image_spec"
    return f"lite.gym.envs.{'.'.join(env_parts)}{suffix}:{_PROVIDER_ATTR}"


def _provider_files() -> Iterable[Path]:
    """``image_spec.py`` files under :data:`_ENVS_ROOT`.

    ``_SKIP_PROVIDER_SEGMENTS`` prunes the descent instead of rejecting paths after
    the fact: most of ``lite/gym/envs`` is ``.cache/``, and an ``rglob`` that
    filters afterwards has already paid to visit all of it.
    """
    for dirpath, dirnames, filenames in os.walk(_ENVS_ROOT):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_PROVIDER_SEGMENTS]
        if "image_spec.py" in filenames:
            yield Path(dirpath, "image_spec.py")


def _discover_provider_paths() -> dict[str, str]:
    """Env-local image specs discoverable by file convention.

    Direct providers live at ``lite/gym/envs/<env_id path>/image_spec.py``.
    Prefix providers can live at ``.../<env_id path>/src/image_spec.py``; the
    provider receives the full requested env id and decides which variants it
    supports. This keeps software-specific naming in the env package instead of
    in this generic utility.
    """
    providers: dict[str, str] = {}
    for path in _provider_files():
        parts = path.relative_to(_ENVS_ROOT).parts
        if len(parts) >= 3 and parts[-2:] == ("src", "image_spec.py"):
            env_parts = parts[:-2]
            env_id = ".".join(env_parts)
            provider = _provider_module_path(env_parts, src=True)
        elif parts[-1] == "image_spec.py":
            env_parts = parts[:-1]
            env_id = ".".join(env_parts)
            provider = _provider_module_path(env_parts, src=False)
        else:
            continue
        existing = providers.get(env_id)
        if existing and existing != provider:
            raise ValueError(
                f"conflicting image freshness providers for {env_id!r}: "
                f"{existing!r} and {provider!r}"
            )
        providers[env_id] = provider
    return dict(sorted(providers.items()))


def _provider_path(env_id: str) -> str:
    parts = tuple(part for part in env_id.split(".") if part)
    for count in range(len(parts), 0, -1):
        env_parts = parts[:count]
        env_dir = _ENVS_ROOT.joinpath(*env_parts)
        has_direct = (env_dir / "image_spec.py").is_file()
        has_src = (env_dir / "src/image_spec.py").is_file()
        if has_direct and has_src:
            raise ValueError(
                f"conflicting image freshness providers for {'.'.join(env_parts)!r}: "
                "both image_spec.py and src/image_spec.py exist"
            )
        if has_direct:
            return _provider_module_path(env_parts, src=False)
        if has_src:
            return _provider_module_path(env_parts, src=True)
    raise UnknownImageFreshnessProvider(
        f"unknown image freshness env_id {env_id!r}; no env-local "
        "image_spec.py provider found"
    )


def _load_provider(path: str) -> ImageProvider:
    module_name, sep, attr = path.partition(":")
    if not sep:
        raise ValueError(f"invalid image freshness provider path: {path!r}")
    module = importlib.import_module(module_name)
    provider = getattr(module, attr)
    if not callable(provider):
        raise TypeError(f"image freshness provider is not callable: {path!r}")
    return provider


def supported_env_ids() -> tuple[str, ...]:
    """Exact env ids accepted by :func:`image_for`.

    Prefix providers may accept additional env-local variants without expanding
    them here; those variants can depend on local material state or env-local
    registration metadata.
    """
    out: set[str] = set()
    for env_id, provider_path in _discover_provider_paths().items():
        module_name, _, _attr = provider_path.partition(":")
        module = importlib.import_module(module_name)
        provider_ids = getattr(module, "supported_env_ids", None)
        if callable(provider_ids):
            raw_ids = provider_ids()
            if isinstance(raw_ids, (str, bytes)) or not isinstance(
                raw_ids, Iterable
            ):
                raise TypeError(
                    "image freshness provider supported_env_ids() must return "
                    f"an iterable of non-empty strings: {provider_path!r}"
                )
            for item in raw_ids:
                if not isinstance(item, str) or not item:
                    raise TypeError(
                        "image freshness provider supported_env_ids() returned "
                        f"invalid env id {item!r}: {provider_path!r}"
                    )
                out.add(item)
        else:
            out.add(env_id)
    return tuple(sorted(out))


def image_for(
    env_id: str,
    tag: str | None = None,
) -> ContainerImage:
    """The ``ContainerImage`` for a container-based env.

    ``tag`` overrides the registry default for envs whose tag is config-driven
    (android_*, osworld, demo read it from YAML). The source hash is tag-independent,
    but the LABEL lives on whatever tag ``install.sh`` builds (the default); if you
    override the tag to one that wasn't built+labeled, the gate will read STALE —
    build+label that tag too. The default YAML tags match the install tags, so this
    only bites a deliberate override.
    """
    provider_path = _provider_path(env_id)
    try:
        img = _load_provider(provider_path)(env_id)
    except KeyError as exc:
        raise ValueError(
            f"image freshness provider {provider_path!r} rejected env_id {env_id!r}"
        ) from exc
    if not isinstance(img, ContainerImage):
        raise TypeError(
            f"image freshness provider {provider_path!r} returned "
            f"{type(img).__name__}, expected ContainerImage"
        )
    return img if (tag is None or tag == img.tag) else replace(img, tag=tag)


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 3 and sys.argv[1] == "hash":
        print(image_for(sys.argv[2]).src_hash())
    else:
        print(
            "usage: python -m lite.gym.utils.backend.freshness "
            "hash <env_id>",
            file=sys.stderr,
        )
        print(
            f"  env_id one of: {', '.join(supported_env_ids())}; "
            "env-local providers may accept additional variants",
            file=sys.stderr,
        )
        sys.exit(1)
