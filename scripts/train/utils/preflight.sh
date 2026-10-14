#!/usr/bin/env bash
#
# Env-server pre-training preflight.
#
# Source from RL launch scripts before training. It fails fast on:
#   1. GET /host_status                 → responsive AND wire-compatible?
#   2. GET /envs/${ENV_ID}              → registered AND available?
#   3. GET/DELETE /instances?session_id=&env_id=
#                                       → prove/release prior-session leftovers
#
# Required caller env:
#   CUA_LITE_ENV_SERVER_URL    e.g. http://<env-server-host>:30100; optional
#                              only for PURE in-process envs. Use localhost
#                              only when client and server share a network namespace.
#   CUA_LITE_ENV_SERVER_TOKEN  bearer
#   ENV_ID                     e.g. androidworld, lite.osworld, browsergym.miniwob
#   SESSION_ID                 batch tag for cleanup scope
#
# Optional policy env:
#   CUA_LITE_EXPECTED_FRAME_MAGIC / CUA_LITE_EXPECTED_FRAME_VERSION
#   CUA_LITE_EXPECTED_COMMIT / CUA_LITE_ALLOW_DIRTY_ENV_SERVER / CUA_LITE_ROOT


# -- Direct-mode gate ----------------------------------------------------------

# Only PURE in-process envs may train without an env-server. For every other
# backend family, lifecycle, reaping, and rollout concurrency belong to the
# server, so refuse here instead of minutes later at the first gym.make.
if [ -z "${CUA_LITE_ENV_SERVER_URL:-}" ]; then
   if command -v uv >/dev/null 2>&1; then
      _PYTHON_CMD=(uv run python)
   else
      _PYTHON_CMD=("${PYTHON:-python3}")
   fi
   _FAMILY=$(ENV_ID="${ENV_ID}" "${_PYTHON_CMD[@]}" -c 'import contextlib, os
from lite.gym.registry import ensure_registered, import_registration_modules
from lite.gym.services import family_of
import_registration_modules()
with contextlib.suppress(Exception): ensure_registered(os.environ["ENV_ID"])
print(getattr(family_of(os.environ["ENV_ID"]), "value", ""))' 2>/dev/null || true)
   if [ "${_FAMILY}" = "pure" ]; then
      echo "DIRECT MODE: ${ENV_ID} has no external backend."
      echo "  Skipping env-server preflight; rollout workers build envs in-process."
      echo "  Export CUA_LITE_ENV_SERVER_URL to train through an env-server."
      unset _FAMILY _PYTHON_CMD
      return 0
   fi
   echo "Error: CUA_LITE_ENV_SERVER_URL is unset for ENV_ID=${ENV_ID}" >&2
   echo "       Resolved backend type:" \
        "${_FAMILY:-<unresolved: env not importable, or no python on PATH>}" >&2
   echo "       Only PURE in-process envs may train without an env-server." >&2
   echo "       Start scripts/serve_env.py and export CUA_LITE_ENV_SERVER_URL +" \
        "CUA_LITE_ENV_SERVER_TOKEN." >&2
   exit 1
fi

# -- Host status / wire compatibility ----------------------------------------

_AUTH="Authorization: Bearer ${CUA_LITE_ENV_SERVER_TOKEN}"
_HOST_URL="${CUA_LITE_ENV_SERVER_URL}/host_status"
_URL="${CUA_LITE_ENV_SERVER_URL}/envs/${ENV_ID}"
_EXPECTED_FRAME_MAGIC="${CUA_LITE_EXPECTED_FRAME_MAGIC:-LEF6}"
_EXPECTED_FRAME_VERSION="${CUA_LITE_EXPECTED_FRAME_VERSION:-6}"
_FRAME_VERSION_RE="\"frame_version\"[[:space:]]*:[[:space:]]*${_EXPECTED_FRAME_VERSION}"
_FRAME_VERSION_RE="${_FRAME_VERSION_RE}([[:space:]]*[,}]|[[:space:]]*$)"
_ROOT="${CUA_LITE_ROOT:-$(pwd)}"
_EXPECTED_COMMIT="$(
   git -C "${_ROOT}" rev-parse HEAD 2>/dev/null || true
)"
_EXPECTED_COMMIT="${CUA_LITE_EXPECTED_COMMIT:-${_EXPECTED_COMMIT}}"

echo "Probing env-server host_status..."
_HOST_RESP=$(curl -sf --max-time 30 -H "${_AUTH}" "${_HOST_URL}") || {
   echo "Error: env-server host_status probe failed for ${_HOST_URL}" >&2
   echo "       Unreachable, unauthorized, or not an env-server." >&2
   exit 1
}
if ! printf '%s' "${_HOST_RESP}" \
   | grep -Eq "\"frame_magic\"[[:space:]]*:[[:space:]]*\"${_EXPECTED_FRAME_MAGIC}\""; then
   echo "Error: env-server wire magic mismatch or missing /host_status.wire.frame_magic" >&2
   echo "       Expected ${_EXPECTED_FRAME_MAGIC}; restart/upgrade serve_env.py." >&2
   exit 1
fi
if ! printf '%s' "${_HOST_RESP}" \
   | grep -Eq "${_FRAME_VERSION_RE}"; then
   echo "Error: env-server wire frame_version mismatch or missing" \
        "/host_status.wire.frame_version" >&2
   echo "       Expected ${_EXPECTED_FRAME_VERSION}; restart/upgrade serve_env.py." >&2
   exit 1
fi
echo "  ok — env-server wire ${_EXPECTED_FRAME_MAGIC}/v${_EXPECTED_FRAME_VERSION}"
if [ -n "${_EXPECTED_COMMIT}" ]; then
   if ! printf '%s' "${_HOST_RESP}" \
      | grep -Eq "\"commit\"[[:space:]]*:[[:space:]]*\"${_EXPECTED_COMMIT}\""; then
      echo "Error: env-server commit mismatch or missing /host_status.cua_lite.commit" >&2
      echo "       Expected ${_EXPECTED_COMMIT}; restart serve_env.py from this checkout." >&2
      exit 1
   fi
fi
if printf '%s' "${_HOST_RESP}" | grep -Eq "\"dirty\"[[:space:]]*:[[:space:]]*true"; then
   if [ "${CUA_LITE_ALLOW_DIRTY_ENV_SERVER:-0}" != "1" ]; then
      echo "Error: env-server reports dirty=true." >&2
      echo "       Commit the server checkout, or set" \
           "CUA_LITE_ALLOW_DIRTY_ENV_SERVER=1 for an explicit dev run." >&2
      exit 1
   fi
   echo "  warning — env-server dirty=true accepted by CUA_LITE_ALLOW_DIRTY_ENV_SERVER=1" >&2
fi

# -- Env availability ---------------------------------------------------------

echo "Probing env-server for env_id='${ENV_ID}'..."
_RESP=$(curl -sf --max-time 30 -H "${_AUTH}" "${_URL}") || {
   echo "Error: env-server probe failed for ${_URL}" >&2
   echo "       Unreachable or env_id not registered. List what IS registered:" >&2
   echo "           curl -H \"\$_AUTH\" \"\${CUA_LITE_ENV_SERVER_URL}/envs\"" >&2
   exit 1
}

# Plain-bash JSON probe — no jq/python needed. Response shape
# (lite.gym.remote.server._env_metadata):
#   {"available": true,  "n_tasks": N, "splits": [...], "env_cost": C}
#   {"available": false, "error": "<install hint>"}
if printf '%s' "${_RESP}" | grep -Eq '"available"[[:space:]]*:[[:space:]]*true'; then
   _N=$(printf '%s' "${_RESP}" \
      | grep -oE '"n_tasks"[[:space:]]*:[[:space:]]*[0-9]+' \
      | grep -oE '[0-9]+$')
   echo "  ok — ${ENV_ID} available (n_tasks=${_N})"
else
   _ERR=$(printf '%s' "${_RESP}" \
      | grep -oE '"error"[[:space:]]*:[[:space:]]*"[^"]*"' \
      | cut -d\" -f4 \
      | head -c 200)
   echo "Error: env_id=${ENV_ID} registered but available=false: ${_ERR}" >&2
   echo "       Run the env's install.sh on the env-server host." >&2
   echo "       See docs/envs.md#installation for env-specific setup." >&2
   exit 1
fi

# -- Prior-session cleanup ----------------------------------------------------

# Release prior-session leftover envs. Query values are urlencoded because
# session ids can contain shell-friendly characters that are special in URLs.
_INST_URL="${CUA_LITE_ENV_SERVER_URL}/instances"
_INST_QUERY=(
   --data-urlencode "session_id=${SESSION_ID}"
   --data-urlencode "env_id=${ENV_ID}"
)
_DRY_CLOSE_RESP=$(
   curl -sf --max-time 30 -X DELETE -G -H "${_AUTH}" \
      "${_INST_QUERY[@]}" \
      --data-urlencode "dry_run=true" \
      "${_INST_URL}"
) || {
   echo "Error: DELETE /instances dry_run failed for session_id=${SESSION_ID} env_id=${ENV_ID}" >&2
   exit 1
}
if printf '%s' "${_DRY_CLOSE_RESP}" | grep -Eq '"would_close"[[:space:]]*:[[:space:]]*\[[^]]'; then
   echo "  closing prior-session leftovers for session_id=${SESSION_ID} env_id=${ENV_ID}"
   _CLOSE_RESP=$(
      curl -sf --max-time 60 -X DELETE -G -H "${_AUTH}" \
         "${_INST_QUERY[@]}" \
         "${_INST_URL}"
   ) || {
      echo "Error: DELETE /instances failed for session_id=${SESSION_ID} env_id=${ENV_ID}" >&2
      exit 1
   }
   if printf '%s' "${_CLOSE_RESP}" \
      | grep -Eq '"skipped_in_flight"[[:space:]]*:[[:space:]]*\[[^]]'; then
      echo "Error: DELETE /instances skipped in-flight instances." >&2
      echo "       Refusing to start rollout into a dirty session." >&2
      echo "       Response: ${_CLOSE_RESP}" >&2
      exit 1
   fi
fi
_LIST_RESP=$(
   curl -sf --max-time 30 -G -H "${_AUTH}" \
      "${_INST_QUERY[@]}" \
      "${_INST_URL}"
) || {
   echo "Error: GET /instances failed after cleanup for" \
        "session_id=${SESSION_ID} env_id=${ENV_ID}" >&2
   exit 1
}
if ! printf '%s' "${_LIST_RESP}" \
   | grep -Eq '"instances"[[:space:]]*:[[:space:]]*\[[[:space:]]*\]'; then
   echo "Error: env-server still has scoped live instances after cleanup." >&2
   echo "       Refusing to start rollout." >&2
   echo "       Response: ${_LIST_RESP}" >&2
   exit 1
fi
echo "  ok — no scoped prior-session instances remain"

unset _AUTH _HOST_URL _URL _HOST_RESP
unset _EXPECTED_FRAME_MAGIC _EXPECTED_FRAME_VERSION _FRAME_VERSION_RE
unset _ROOT _EXPECTED_COMMIT
unset _RESP _N _ERR _INST_URL _INST_QUERY _DRY_CLOSE_RESP _CLOSE_RESP _LIST_RESP
