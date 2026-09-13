#!/usr/bin/env python3
"""Launcher for the cua-lite env server (FastAPI + uvicorn).

Operator-facing deployment policy lives here: CLI flags and auth mode.
Admission, wire, and state logic live under :mod:`lite.gym.remote`; this file
just builds config and hands off.

For the architecture overview and when-to-use guidance, see
[docs/envs.md#env-server](/docs/envs.md#env-server).
For per-flag reference, see ``--help``.

Usage::

    # Default: passthrough auth (any bearer accepted, identity by hash)
    uv run python scripts/serve_env.py --port 30100

    # Strict auth (single bearer; rotate to invalidate every client)
    uv run python scripts/serve_env.py --port 30100 --token <T>

    # Two instances on one host (separate ports, separate token scopes)
    uv run python scripts/serve_env.py --port 30100 --token alice &
    uv run python scripts/serve_env.py --port 30101 --token bob &

Install env-specific host dependencies with that env's ``install.sh`` first;
then launch this server with regular ``uv run python``.

Client env vars (read by :mod:`lite.gym.remote.client`)::

    CUA_LITE_ENV_SERVER_URL    — e.g. http://host:30100
    CUA_LITE_ENV_SERVER_TOKEN  — --token value in strict mode; any bearer in passthrough
"""

from __future__ import annotations

import argparse
import logging

import uvicorn

from lite.gym.remote import State, make_app
from lite.gym.remote.admission import (
    AdmissionGate,
    derive_admission_config,
    log_admission_config,
)
from lite.gym.remote.alive import (
    log_keep_alive_timeout,
    resolve_keep_alive_timeout,
)
from lite.gym.utils.server.capacity import cached_host_capacity
from lite.utils.logging import setup_logging


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="cua-lite env server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=30100)
    parser.add_argument(
        "--max-live-envs",
        type=int,
        default=None,
        help=(
            "Maximum concurrent live env instances. Default auto-scales from "
            "host capacity; single-tenant high-throughput servers can raise it."
        ),
    )
    parser.add_argument(
        "--idle-ttl-sec",
        type=float,
        default=600.0,
        help="Auto-close envs idle longer than this. Default 600s.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help=(
            "Bearer token. Unset = PASSTHROUGH (any bearer; identity = sha256 "
            "of bearer). Set = STRICT."
        ),
    )
    parser.add_argument(
        "--admin-token",
        default=None,
        help=(
            "Bearer for /admin/* endpoints. Unset in passthrough = admin OPEN; "
            "unset in strict = admin DISABLED (404)."
        ),
    )
    parser.add_argument(
        "--env-ids",
        nargs="+",
        default=None,
        metavar="ID",
        help="Allow-list of env_ids. Unset = all accepted.",
    )
    parser.add_argument(
        "--warm-singleton",
        action="store_true",
        dest="warm_singleton",
        help=(
            "Background pre-warm every served SINGLETON backend at startup. "
            "The server listens immediately; warming envs return retriable 503. "
            "Pair with --env-ids to scope the warm set."
        ),
    )
    parser.add_argument(
        "--timeout-keep-alive",
        type=float,
        default=None,
        dest="timeout_keep_alive",
        help=(
            "HTTP idle keep-alive timeout, seconds. Default is derived to "
            "stay above the client keep-alive window."
        ),
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Build env-server config and hand off to uvicorn."""
    setup_logging()
    logger = logging.getLogger("cua-lite.env-server")

    overrides: dict = {}
    if args.max_live_envs is not None:
        overrides["max_live_envs"] = args.max_live_envs

    host = cached_host_capacity()
    admission_cfg = derive_admission_config(host, **overrides)
    admission_gate = AdmissionGate(admission_cfg)
    log_admission_config(
        admission_cfg,
        explicit_max_live_envs=args.max_live_envs is not None,
    )

    allowed = set(args.env_ids) if args.env_ids is not None else None
    if args.token is None:
        logger.warning(
            "AUTH passthrough: any bearer accepted (host=%s, admin_token=%s). "
            "Bind 127.0.0.1 or set --token+--admin-token for prod.",
            args.host, "set" if args.admin_token else "unset",
        )
    else:
        logger.info("AUTH: strict mode.")

    state = State.for_env_server(
        admission=admission_gate,
        idle_ttl_sec=args.idle_ttl_sec,
        allowed_env_ids=allowed,
        warm_singleton_enabled=args.warm_singleton,
    )
    logger.info(
        "env server: host=%s port=%d env_ids=%s idle_ttl=%.0fs "
        "reset_concurrency=%d warm_singleton=%s",
        args.host,
        args.port,
        sorted(allowed) if allowed is not None else "ALL",
        args.idle_ttl_sec,
        state.reset_concurrency,
        "on" if args.warm_singleton else "off",
    )

    keep_alive = resolve_keep_alive_timeout(args.timeout_keep_alive)
    log_keep_alive_timeout(
        logger,
        keep_alive,
        explicit=args.timeout_keep_alive is not None,
    )
    uvicorn.run(
        make_app(state, token=args.token, admin_token=args.admin_token, port=args.port),
        host=args.host, port=args.port, log_level="info",
        timeout_keep_alive=keep_alive,
    )


if __name__ == "__main__":
    main(_parse_args())
