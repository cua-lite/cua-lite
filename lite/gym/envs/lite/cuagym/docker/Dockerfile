# lite.cuagym image — ONE image for CUA-Gym browser-side + desktop-shaped tasks.
#
# Both task families run on the same lite.osworld desktop substrate (Xvnc :1 +
# Sandbox exec-stdio server + Chrome + LibreOffice/VSCode/... + a baked
# /home/user/.Xauthority). Their only difference is a small, additive, non-
# conflicting set of extras, so we bake BOTH into one image rather than maintain
# two:
#   - desktop: the python doc/PDF/image libs the CUA-Gym desktop setup/reward
#     scripts import;
#   - browser side: Node 20 (to serve each mock via `vite preview`), one shared
#     node_modules (the union of every mock's deps), the pre-built mocks under
#     /opt/mocks, and a google-chrome wrapper that opens maximised for a
#     near-fullscreen first frame.
#
# A browser task carries the (unused) doc libs and a desktop task carries the
# (unused) node+mocks — a few hundred MB each, negligible against the ~5 GB
# osworld base, and the base layers are shared with it on disk.
#
# The mocks/ + mocks-package.json build context is staged by scripts/install.sh.
#
# Build (via scripts/install.sh):
#   docker build -t cua-lite/lite.cuagym:latest -f .../cuagym/docker/Dockerfile <ctx>
# For private/throwaway validation, scripts/install.sh can pass
#   --build-arg BASE_IMAGE=cua-lite/lite.osworld:<tag>
# --- builder stage: compile xcftools (xcf2png) from source. 9 GIMP reward
#     scripts call `xcf2png` to rasterize a .xcf → PNG for pixel checks, but
#     xcftools was dropped from the Ubuntu 22.04 repos (unmaintained). Compile it
#     here and COPY only the 2 binaries into the final image, so the build
#     toolchain never bloats it (runtime libpng16 is already in the desktop base).
ARG BASE_IMAGE=cua-lite/lite.osworld:latest
FROM ${BASE_IMAGE} AS xcftools-builder
USER root
ARG XCFTOOLS_REV=be488bd2e35a0c6794141411cb5d290950ba4345
RUN DEBIAN_FRONTEND=noninteractive apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential libpng-dev perl gettext autoconf automake autopoint pkg-config git ca-certificates \
 && git clone https://github.com/j-jorge/xcftools /tmp/xcftools \
 && cd /tmp/xcftools && git checkout "$XCFTOOLS_REV" \
 && cd /tmp/xcftools && autoreconf -i && ./configure && make \
 && strip xcf2png xcf2pnm

FROM ${BASE_IMAGE}

USER root

# CUA-Gym launches VLC directly instead of through OSWorld's dispatcher, so its
# child image owns the first-run state needed by those tasks.
RUN mkdir -p /home/user/.config/vlc && \
    printf '[qt]\nqt-privacy-ask=0\n[core]\nmetadata-network-access=0\n' \
      > /home/user/.config/vlc/vlcrc && \
    chown -R user:user /home/user/.config/vlc

# Keep the shared lite.osworld desktop appearance unchanged. Append only the
# child readiness marker after the inherited settings have been applied.
RUN cat >> /usr/local/bin/apply-settings.sh <<'EOF'
touch /tmp/lite-cuagym-ready
EOF

# xcftools is used only by reward scripts. Keep it on the env-side PATH rather
# than exposing tools absent from the OSWorld guest to the agent shell.
COPY --from=xcftools-builder /tmp/xcftools/xcf2png /tmp/xcftools/xcf2pnm /opt/env/bin/

# --- desktop: doc/PDF/image libs the CUA-Gym desktop scripts import (scanned
#     from all desktop bundles; the base image lacks them). seaborn/gymnasium/
#     fastapi/pycryptodome added from a full-dataset static import scan
#     (pycryptodome is PyPDF2's runtime AES-256 backend, not a direct import).
#
#     ENV_PY (/opt/env/venv) ONLY — never the agent's /usr/bin/python3. cuagym has
#     that split (reward/setup run on the 3.12 venv, the agent sees the 3.10 system
#     python), so anything added here is invisible to the agent. Keep it that way:
#     cuaworld, which lacks the split, had to weigh agent-visibility to close the
#     same class of gap.
#
#     opencv-python-headless / pdfminer.six / toml close the remaining ENV_PY gaps
#     found by an AST scan of all 10910 pinned bundles (9405 desktop + 1505 web),
#     probed against this image's own /opt/env/venv/bin/python:
#       cv2       5 reward.py (4e9315a1, 9477a761, c72361e9, fab83ca5, 9d7562e7)
#       pdfminer  3 reward.py (883c7075, ad8e7cee, c0322053) — `pdfminer.high_level`
#       toml      1 reward.py (d99d6453)
#     All on the REWARD path, all try/excepted upstream, so each was scoring 0 for an
#     environment reason — the `reward_judge` shape exactly. They must be PINNED here
#     even though an older build of this image happened to resolve them transitively:
#     that was an accident of the resolver, it did not survive a rebuild, and a
#     dependency you did not ask for is not a dependency you have. headless, not
#     plain opencv-python: no GUI libs are wanted in a reward interpreter.
#
#     With those three, the scan leaves exactly four unsatisfied imports, and NONE of
#     them wants a line here — see `_KNOWN_MISSING` in
#     tests/gym/envs/lite/cuagym/test_cuagym_bundle_imports.py, which records rather than
#     tolerates them:
#       torch + stable_baselines3  2 reward.py (57e6a588 wants both, 392066aa torch).
#         ~2.5 GB onto a 6.7 GB image for 2 of 10910 bundles. Skipped on cost.
#       pyods  1 initial_setup.py (a5ed1cb4) — DEAD import: `try: import pyods /
#         except ImportError: pass`, after which the function builds the .ods with
#         odfpy, which IS installed. Nothing is lost.
#       pip    3 reward.py + many setup scripts. Handled below by an env-side
#         pip shim, not by exposing system pip: bare `pip`/`pip3` and
#         `python3 -m pip` inside setup/reward sessions resolve to /opt/env/venv,
#         while upstream's VM-style `--user` flag is treated as compatibility
#         noise and stripped before invoking uv.
#     UNO_PY (/opt/env/uno-venv) has NO gap either: `com` looks missing to find_spec,
#     but `import uno; from com.sun.star.beans import PropertyValue` works there —
#     the names resolve through uno.py's import hook, not a `com` package. ---
RUN /opt/env/bin/uv pip install --python /opt/env/venv/bin/python --no-cache-dir \
        openpyxl==3.1.5 python-pptx==1.0.2 python-docx==1.1.2 \
        PyMuPDF==1.25.5 odfpy==1.4.1 reportlab==5.0.0 PyPDF2==3.0.1 \
        pikepdf==10.9.1 fpdf2==2.8.7 pypdf==6.14.2 \
        gimpformats==2025 ezodf==0.3.2 pyexcel-ods3==0.6.1 \
        msoffcrypto-tool==6.0.0 mutagen==1.47.0 piexif==1.1.3 pyHanko==0.35.2 \
        matplotlib==3.11.0 scipy==1.18.0 scikit-image==0.26.0 pytest==9.1.1 \
        pandas==3.0.3 dbf==0.99.11 seaborn==0.13.2 gymnasium==1.3.0 \
        fastapi==0.139.0 pycryptodome==3.23.0 websocket-client==1.9.0 \
        playwright==1.60.0 \
        opencv-python-headless==4.13.0.90 pdfminer.six==20251107 toml==0.10.2

# Passwordless sudo for the desktop user. 109 desktop setup scripts escalate as
# `echo 'password' | sudo -S ...` and another 72 call bare `sudo ...`; NONE uses the
# `echo user | sudo -S` idiom the parent's password-gated account expects (that idiom
# is the OSWorld corpus's, and this corpus is not it). Bare `sudo` in a non-interactive
# script only ever worked if the authoring VM was passwordless, so this restores what
# those 181 bundles were written against instead of failing them at reset.
#
# Scoped to THIS image: the parent stays password-gated (`user:user`) for lite.osworld
# and lite.scalecua, whose own generated tasks type that password. No capability is
# added either way -- the account's password is its username, in plain sight.
RUN printf 'user ALL=(ALL) NOPASSWD: ALL\n' > /etc/sudoers.d/99-cuagym-nopasswd \
 && chmod 0440 /etc/sudoers.d/99-cuagym-nopasswd

# A VM-parity `systemctl`, the same idiom the parent uses for `timedatectl` /
# `hostname` / `powerprofilesctl`: "the OSWorld VM exposes systemd's ...; this
# container does not run systemd, but tasks still expect the VM-shaped command
# output. Install a shim so setup does not rewrite shell scripts on every reset."
#
# 738 desktop bundles drive services through systemctl. Real /usr/bin/systemctl is
# the systemd binary and answers `System has not been booted with systemd as init
# system (PID 1). Can't operate.` with rc=1, so `sudo systemctl restart nginx`
# killed setup outright once passwordless sudo let it get that far.
#
# The packages these bundles install still ship SysV init scripts, and those work
# here (verified: `service nginx start` -> rc=0, nginx serving). So map the verbs
# onto /etc/init.d and answer the unit-state queries from the process table.
# Env PATH only, so the agent's own `systemctl` keeps its real VM-shaped failure.
RUN cat > /opt/env/bin/systemctl <<'SYSCTL' && chmod 0755 /opt/env/bin/systemctl
#!/usr/bin/env bash
# CUA-Gym env-side systemctl shim for upstream setup/reward scripts. See Dockerfile.
set -u
verb="${1:-}"; shift || true
# Unit-file bookkeeping has no meaning without systemd, and every caller only cares
# that it succeeded.
case "$verb" in
  daemon-reload|daemon-reexec|enable|disable|mask|unmask|preset) exit 0 ;;
esac
unit="${1:-}"; unit="${unit%.service}"
[ -z "$unit" ] && exit 0
case "$verb" in
  is-active)  pgrep -x "$unit" >/dev/null && { echo active; exit 0; }; echo inactive; exit 3 ;;
  is-enabled) echo enabled; exit 0 ;;
  status)     service "$unit" status; exit $? ;;
  start|stop|restart|reload|force-reload|try-restart)
    [ "$verb" = force-reload ] && verb=restart
    [ "$verb" = try-restart ] && verb=restart
    if [ -x "/etc/init.d/$unit" ]; then service "$unit" "$verb"; exit $?; fi
    # No init script: starting is a no-op the caller cannot act on, but stopping
    # something that is running is still expected to work.
    case "$verb" in stop) pkill -x "$unit" >/dev/null 2>&1; exit 0 ;; *) exit 0 ;; esac ;;
esac
exit 0
SYSCTL

# CUA-Gym shell setup scripts inherit exec-stdio's env PATH. Make their common
# VM idioms env-side too:
#   - `pip`, `pip3`, and `python3 -m pip` install into /opt/env/venv via uv.
#   - `--user` / `--break-system-packages` are ignored compatibility flags.
#   - a few scripts probe `python-docx` as `import python_docx` (same for pptx);
#     provide tiny aliases so baked dependencies do not trigger live installs.
RUN /opt/env/venv/bin/python - <<'PY'
from pathlib import Path
import site

site_dir = Path(site.getsitepackages()[0])
(site_dir / "pip.py").write_text(
    '''"""CUA-Gym env-side pip shim for upstream setup/reward scripts."""

from __future__ import annotations

import subprocess
import sys

_UV = "/opt/env/bin/uv"
_PYTHON = "/opt/env/venv/bin/python"
_DROP = {
    "--user",
    "--break-system-packages",
    "--disable-pip-version-check",
    "--no-input",
    "--noinput",
    "--no-warn-script-location",
}
_DROP_WITH_VALUE = {"--root-user-action", "--timeout", "--retries"}
_TRANSLATE = {"--no-cache-dir": "--no-cache", "--constraint": "--constraints"}
_SELF_REQUIREMENTS = {"pip"}
_TARGET_WITH_VALUE = {"-r", "--requirements", "-e", "--editable"}
_VALUE_ONLY_WITH_VALUE = {
    "-c",
    "--constraints",
    "--overrides",
    "-b",
    "--build-constraints",
    "--index-url",
    "--extra-index-url",
    "-f",
    "--find-links",
}


def _clean_install_args(args: list[str]) -> list[str]:
    cleaned: list[str] = []
    skip_next = False
    for arg in args:
        if skip_next:
            skip_next = False
            continue
        if arg in _DROP:
            continue
        if arg in _DROP_WITH_VALUE:
            skip_next = True
            continue
        if any(arg.startswith(prefix + "=") for prefix in _DROP_WITH_VALUE):
            continue
        if arg in _TRANSLATE:
            cleaned.append(_TRANSLATE[arg])
            continue
        translated = False
        for old, new in _TRANSLATE.items():
            if arg.startswith(old + "="):
                cleaned.append(new + arg[len(old):])
                translated = True
                break
        if translated:
            continue
        if _is_self_requirement(arg):
            continue
        cleaned.append(arg)
    return cleaned


def _is_self_requirement(arg: str) -> bool:
    if arg.startswith("-"):
        return False
    normalized = arg.split("[", 1)[0].lower()
    for marker in ("==", ">=", "<=", "~=", "!=", ">", "<"):
        normalized = normalized.split(marker, 1)[0]
    return normalized in _SELF_REQUIREMENTS


def _has_install_target(args: list[str]) -> bool:
    skip_next = False
    for arg in args:
        if skip_next:
            skip_next = False
            continue
        if arg in _TARGET_WITH_VALUE:
            return True
        if any(arg.startswith(prefix + "=") for prefix in _TARGET_WITH_VALUE):
            return True
        if arg in _VALUE_ONLY_WITH_VALUE:
            skip_next = True
            continue
        if any(arg.startswith(prefix + "=") for prefix in _VALUE_ONLY_WITH_VALUE):
            continue
        if not arg.startswith("-"):
            return True
    return False


def main(args: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if args is None else args)
    if not argv or argv[0] in {"-h", "--help"}:
        print("CUA-Gym pip shim: installs into /opt/env/venv via uv.")
        return 0
    if argv[0] in {"-V", "--version"}:
        print("pip shim for /opt/env/venv")
        return 0
    command, *rest = argv
    if command == "install":
        rest = _clean_install_args(rest)
        if not _has_install_target(rest):
            return 0
    cmd = [_UV, "pip", command, "--python", _PYTHON, *rest]
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    raise SystemExit(main())
'''
)
(site_dir / "python_docx.py").write_text("from docx import *  # noqa: F401,F403\n")
(site_dir / "python_pptx.py").write_text("from pptx import *  # noqa: F401,F403\n")
for name in ("pip", "pip3", "pip3.12"):
    path = Path("/opt/env/venv/bin") / name
    path.write_text("#!/usr/bin/env bash\nexec /opt/env/venv/bin/python -m pip \"$@\"\n")
    path.chmod(0o755)
PY

# The apt half of the same story. 130 desktop setup scripts `apt-get install` a tool
# the source VM lacked; the parent strips /var/lib/apt/lists, so real apt answers
# `E: Unable to locate package` (rc=100) and the script dies before turn_00.
#
# Two things must both hold, which is why this is a shim and not just restoring the
# lists. Restoring them alone would let `apt-get install imagemagick|pandoc|tesseract-ocr`
# SUCCEED and re-register convert/pandoc/tesseract on /usr/bin -- the agent-PATH re-leak
# the parent's relocate step forbids, happening per task at reset, silently.
#
#   * Already provided (the image bakes every high-frequency package, most of them
#     relocated to /opt/env/bin so dpkg no longer records them): exit 0, no network.
#     This is where every CLI_FORBIDDEN name lands, so the leak path is never taken.
#   * Anything else: refresh the lists ONCE per container, then run the real install so
#     the package actually arrives, and re-run the relocate step afterwards in case a
#     dependency dragged a forbidden CLI back onto /usr/bin. The refresh is lazy -- only
#     the tasks that ask for an unbaked package pay it, not every reset.
#
# Env PATH only (/opt/env/bin), so the agent's own `apt-get` is untouched.
RUN cat > /opt/env/bin/apt-get <<'SHIM' && chmod 0755 /opt/env/bin/apt-get \
 && ln -sf /opt/env/bin/apt-get /opt/env/bin/apt
#!/usr/bin/env bash
# CUA-Gym env-side apt shim for upstream setup/reward scripts. See Dockerfile.
set -u
real=/usr/bin/apt-get
[ "$(basename "$0")" = apt ] && real=/usr/bin/apt
[ "${1:-}" != install ] && exec "$real" "$@"

# Packages whose command the image provides from /opt/env/bin after the relocate
# step, so dpkg no longer records them. Closed set: the parent's relocate list plus
# this image's own. Every CLI_FORBIDDEN name is here.
VENDORED=" imagemagick imagemagick-6.q16 xdotool xclip jq pandoc pdftk pdftk-java \
 tesseract-ocr tesseract-ocr-eng sox sqlite3 cmake "
HIDDEN="xdotool convert import mogrify identify compare composite montage animate conjure display stream \
 xclip xsel jq pandoc pdftk tesseract sqlite3 sox cmake"

missing=()
for a in "$@"; do
  case "$a" in install|-*) continue ;; esac
  dpkg-query -W "$a" >/dev/null 2>&1 && continue
  [[ "$VENDORED" == *" $a "* ]] && continue
  missing+=("$a")
done
[ ${#missing[@]} -eq 0 ] && exit 0

# Unbaked package: make the install actually work. The image bakes the apt indices, so
# this normally needs no network -- refresh only if something emptied them, and only once.
[ -n "$(ls -A /var/lib/apt/lists 2>/dev/null | grep -v '^lock$')" ] || [ -f /var/lib/lite/apt-refreshed ] || {
  mkdir -p /var/lib/lite
  DEBIAN_FRONTEND=noninteractive "$real" update >/dev/null 2>&1 && touch /var/lib/lite/apt-refreshed
}
shift
DEBIAN_FRONTEND=noninteractive "$real" install "$@"
rc=$?
sh /opt/lite/relocate-clis.sh $HIDDEN >/dev/null 2>&1 || true
exit $rc
SHIM

# --- desktop: the parent already owns gedit and the system-Python UNO bridge.
#     Assert those inherited dependencies rather than reinstalling/upgrading
#     base-pinned packages. Add only what CUA-Gym needs beyond the parent:
#     LibreOffice Base + embedded HSQLDB support, OpenSSH clients, and
#     rustc/cargo. OpenSSH and Rust stay agent-facing because official tasks
#     explicitly require ssh/scp and Rust commands. Base is required by .odb,
#     bibliography, registered data source, and mail-merge tasks. ---
RUN dpkg-query -W gedit python3-uno >/dev/null \
 && DEBIAN_FRONTEND=noninteractive apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends --no-upgrade \
      libreoffice-base libreoffice-sdbc-hsqldb \
      openssh-client rustc cargo \
 && rm -rf /var/lib/apt/lists/*

# --- The four CLIs a setup script asks apt for that the parent does NOT already
#     vendor. Everything else its `apt-get install` lines name is present already:
#     convert/pandoc/jq/pdftotext/gs/node/ffmpeg/gimp/vlc/curl/gpg/java resolve on the
#     env PATH, so those installs were only ever re-fetching what is here.
#
#     `relocate-clis.sh` MUST run after this install, exactly as the parent's own
#     comment requires: tesseract, sqlite3 and sox are in the parity gate's
#     CLI_FORBIDDEN, so leaving them on /usr/bin would re-leak them onto the agent
#     PATH and change what the agent can do. cmake is not forbidden but is not in
#     the OSWorld guest either, so it goes env-side for the same parity reason. ---
RUN DEBIAN_FRONTEND=noninteractive apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      tesseract-ocr sqlite3 sox cmake \
 && sh /opt/lite/relocate-clis.sh tesseract sqlite3 sox cmake \
 && rm -rf /var/lib/apt/lists/*

# A few official setup scripts need both LibreOffice's Python-3.10-only UNO
# bridge and document libraries. Keep that combination env-facing under
# /opt/env rather than installing those packages into the agent's system Python.
RUN /opt/env/bin/uv venv --python /usr/bin/python3 --system-site-packages \
      /opt/env/uno-venv \
 && /opt/env/bin/uv pip install --python /opt/env/uno-venv/bin/python --no-cache-dir \
      openpyxl==3.1.5 python-docx==1.1.2 odfpy==1.4.1 \
 && chown -R user:user /home/user/.cache/uv

# The uv steps above run as root but inherit HOME=/home/user, so uv writes its
# INTERPRETER cache (`~/.cache/uv/interpreter-v4/<hash>/`) into the desktop user's
# home as root — `--no-cache-dir` suppresses only the PACKAGE cache. The pip shim
# above routes every `pip install` from a setup/reward script through uv as `user`,
# and uv refreshes that same entry, so a root-owned directory makes those scripts
# die with `Permission denied ... interpreter-v4/<hash>/.tmpXXXX` (rc=2 → a
# CuaGymTaskError setup failure). 1006 of the 9405 desktop bundles install packages
# at setup, so the chown above is what keeps that tenth of the split runnable.


# --- browser side: Node 20 (NodeSource) to serve the mocks via `vite preview` ---
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends nodejs \
 && rm -rf /var/lib/apt/lists/*

# Restore the parent's baked apt indices. This image's own apt RUNs each end with the
# standard `rm -rf /var/lib/apt/lists/*`, which silently undid lite.osworld's deliberate
# decision to KEEP them (see its Dockerfile: "the osworld VM guest ships populated
# indices ... VM-faithful"). With them gone, `apt-get install <pkg>` answered
# `E: Unable to locate package` for BOTH callers: a setup script at reset, and the AGENT
# itself typing `sudo apt-get install ...` in the guest terminal -- something the real VM
# allows and lite.osworld already allows. ~40-60 MB, and it keeps the reset critical path
# free of network I/O.
#
# This does NOT reopen the agent-PATH leak: the env-side apt shim above short-circuits
# every vendored package before real apt sees it, and re-runs the relocate step on the
# installs that do go through.
RUN DEBIAN_FRONTEND=noninteractive apt-get update -qq

# Re-hide the parent's JRE CLIs. The parent removes java's siblings by walking every
# /usr/bin symlink that resolves into /usr/lib/jvm, and warns that its step "MUST stay
# after every apt install, or a later install could re-register the CLI on /usr/bin and
# re-leak it onto the agent PATH". This image's apt RUNs sit after that step, and one of
# them pulled an openjdk-11-jre point upgrade whose postinst re-ran
# `update-alternatives --install` -- putting java/keytool/rmiregistry/jjs/jexec/pack200/
# unpack200/rmid back on the agent PATH. `--no-upgrade` does not help: it constrains the
# packages NAMED on the command line, not their dependencies.
#
# Same walk as the parent, so the set stays complete-by-construction rather than a list
# that drifts, plus `java` itself (the parent vendors it to /opt/env/bin, which is where
# LibreOffice and pdftk find it).
RUN for l in /usr/bin/*; do [ -L "$l" ] || continue; \
        case "$(readlink -f "$l" 2>/dev/null)" in \
          /usr/lib/jvm/*) c="$(basename "$l")"; \
                          update-alternatives --remove-all "$c" >/dev/null 2>&1 || true; \
                          rm -f "$l" ;; \
        esac; \
    done \
 && chmod -R a+rX /opt/env


# --- browser side: one shared node_modules (union of mock deps) + the pre-built mocks ---
COPY --chmod=0644 mocks-package.json /opt/mocks/package.json
RUN chmod 0755 /opt/mocks \
 && cd /opt/mocks \
 && npm install --no-audit --no-fund \
 && npm cache clean --force
COPY --chown=user:user mocks/ /opt/mocks/

# --- browser-side evaluator tools. /opt/env/bin is prepended only for env-side exec-stdio
#     commands. Keep an env-side google-chrome forwarder there for setup scripts,
#     but route it to the agent-facing cuagym wrapper below so bare
#     `google-chrome <url>` still opens maximised with CDP on :9222.
COPY --chmod=0755 page_health.py /opt/env/bin/lite-cuagym-page-health
RUN printf '#!/bin/bash\nexec /usr/local/bin/google-chrome "$@"\n' \
      > /opt/env/bin/google-chrome \
 && chmod +x /opt/env/bin/google-chrome

# CUA-Gym uses exec-stdio directly and never calls the inherited OSWorld HTTP
# server. Do not start an unused service in every task container.
RUN rm -f /etc/supervisor/conf.d/50-osworld-server.conf

# --- desktop: VSCode (Electron) refuses the Chromium sandbox in a container
#     ("Failed to move to new namespace ... Operation not permitted"). The official
#     CUA-Gym scripts launch `code <dir>` as-is, so shadow `code` (earlier on PATH
#     than /usr/bin/code) with a --no-sandbox wrapper. Scrub PATH first because
#     VS Code integrated terminals inherit the GUI process environment. ---
# (--disable-workspace-trust skips VSCode's "trust the authors?" modal that would
# otherwise block the editor on every task's first frame.)
RUN printf '#!/bin/bash\nexport PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\nexec /usr/share/code/bin/code --no-sandbox --disable-workspace-trust "$@"\n' \
      > /usr/local/bin/code \
 && chmod +x /usr/local/bin/code \
 && rm -f /usr/bin/code \
 && install -m 0755 /usr/local/bin/code /usr/bin/code

# --- desktop: two browsers/viewers a few tasks launch that can't be apt-installed
#     cleanly here, aliased to an equivalent already in the image:
#       chromium (29 tasks) — Ubuntu 22.04 only ships it as a snap shim (needs
#         snapd, broken in a container) → same Blink engine as google-chrome;
#       okular   (3 tasks)  — real install pulls ~220 KDE packages → evince is
#         the image's PDF viewer.
RUN ln -sf /usr/local/bin/google-chrome /usr/local/bin/chromium \
 && ln -sf /usr/bin/evince /usr/local/bin/okular

# cuagym's agent-facing maximised + CDP(:9222) google-chrome wrapper overrides the base
# osworld chrome on /usr/local/bin. The env-side /opt/env/bin/google-chrome forwarder
# above delegates here, so setup scripts and the agent use the same browser behavior
# without putting /opt/env/bin on the agent's login PATH.
COPY --chmod=0755 google-chrome /usr/local/bin/google-chrome
