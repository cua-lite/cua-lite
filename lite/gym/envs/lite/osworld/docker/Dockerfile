# syntax=docker/dockerfile:1.4
# ^ REQUIRED. Forces BuildKit to use the v1.4 frontend.
#
# Split rule:
#   - docker/{bin,desktop,fonts,runtime,server,software,supervisor}/ contains
#     static runtime artifacts that are copied into the image as files.
#   - Dockerfile keeps build transactions whose order is semantically load-bearing:
#     apt/uv installs, vendoring agent-excluded CLIs into /opt/env/bin followed by
#     `apt-get remove` de-registration, Python boundary management, labels,
#     permissions, and image freshness.
#
# `software` is intentionally singular: it is the app-configuration category,
# not a plural of individual software packages.
#
# lite.osworld additive image — LibreOffice + GIMP + VLC + Thunderbird + Chrome
# + VS Code + OSWorld Flask server, layered on top of the GNOME-Shell base.
#
# In-image server:
#   - OSWorld Flask server (:5000) — setup + evaluation (HTTP: /execute, /file, /accessibility) [this file]
# Host↔container interaction:
#   - exec-stdio (lite/gym/sandbox/exec_stdio/) — one persistent
#     ``docker exec -i`` JSON-lines session, no published agent port. Bakes
#     server.py at /opt/lite/stdio_server.py (baked by the shared base).
#
# Build (context MUST be the env's osworld/ root, NOT docker/, because
# the Dockerfile does `COPY docker/...` and needs the parent):
#   uv run --no-sync bash lite/gym/envs/lite/osworld/scripts/install.sh           # build if missing/stale
#   uv run --no-sync bash lite/gym/envs/lite/osworld/scripts/install.sh rebuild   # force
# Manual equivalent (two-stage):
#   # Stage 1: the SHARED desktop base (GNOME-Shell + Xvnc/noVNC)
#   bash lite/gym/sandbox/scripts/install.sh          # builds cua-lite/sandbox.linux:latest
#   # Stage 2: build additive (apps + OSWorld server + appearance), FROM the base
#   docker build -t cua-lite/lite.osworld:latest \
#     -f lite/gym/envs/lite/osworld/docker/Dockerfile \
#     lite/gym/envs/lite/osworld/
#
# Run (standalone test):
#   docker run -d --name test --memory 4g --cpus 1 \
#     -p 8101:6901 -p 5100:5000 \
#     -e VNC_PW=password -e VNCOPTIONS=-disableBasicAuth \
#     -e VNC_RESOLUTION=1920x1080 \
#     cua-lite/lite.osworld:latest
#   # OSWorld server on localhost:5100, VNC on localhost:8101. No :8000 — the
#   # cua computer-server is fully retired (not present in base or additive);
#   # host↔container ops ride exec-stdio over ``docker exec``, which publishes
#   # no host port.
#
# Pinning policy: every apt package version-pinned (versions taken from the
# previous cua-lite/lite.osworld:latest image at the parity baseline).

ARG BASE_TAG=latest
FROM cua-lite/sandbox.linux:${BASE_TAG}

# ===========================================================================
# Base fill-ins — things the retired lite.osworld.base provided that the shared
# sandbox base does NOT. Kept osworld-specific (task tooling / oracle runtime),
# so the shared base stays generic. (python3.12 + uv + the env venv are now
# provided by the base itself — see sandbox.linux Dockerfile — so osworld only
# EXTENDS the venv and USES the base's uv below.)
# ===========================================================================

# Missing system packages: Chrome's apt repo needs gpg; osworld runtime/oracle
# assume wmctrl (open_file), pulseaudio (audio tasks), ghostscript (`gs` in 90+
# oracle_actions), file, net tools; libxcb/xkbcommon are Qt/XCB runtime shims.
RUN DEBIAN_FRONTEND=noninteractive apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        unzip zip gcc isc-dhcp-client \
        libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 \
        libxcb-xinerama0 libxcb-shape0 libxcb-randr0 libxcb-xfixes0 \
        libxcb-sync1 libxcb-util1 libxcb-cursor0 libxkbcommon-x11-0 \
        software-properties-common net-tools netcat socat wmctrl \
        gpg gpg-agent dirmngr ghostscript file \
        pulseaudio-utils pulseaudio \
        bc \
        xdg-desktop-portal xdg-desktop-portal-gtk \
    && rm -rf /var/lib/apt/lists/*
# xdg-desktop-portal(-gtk): the osworld reference renders GTK file-chooser dialogs
# (VS Code/Electron Open/Save) with the Yaru GREEN suggested-action accept button;
# lite showed a GREY button because, without the GTK portal, Electron fell back to
# its own chooser that doesn't pick up the live Yaru accent (verified in the
# head-to-head). Installing the portal routes Electron dialogs through real GTK.
# `bc` is present in the osworld guest but was missing here; an eval checker that
# does `echo "$x>=1.2" | bc` returned empty → reward 0 on an otherwise-correct
# text-scaling task (unwinnable on lite without it).

# Build tools + Python dev headers so `uv pip` can compile sdists (evdev for
# pynput, and any on-the-fly eval pip).
RUN DEBIAN_FRONTEND=noninteractive apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        libncursesw5-dev libssl-dev libsqlite3-dev tk-dev \
        libgl1-mesa-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Env python 3.12 (uv-managed, isolated) + venv (/opt/env/venv) + `uv` are now OWNED BY THE
# BASE (sandbox.linux) — the generic env-python foundation, distinct from the agent's system
# 3.10 (guest mirror). osworld does NOT install them; it
# only EXTENDS the base's /opt/env/venv (below) with its office/setup/eval libs, and uses
# the base's `uv` to install the agent guest-mirror set into the system 3.10. Build-time
# AND runtime `python3` = /usr/bin/python3 = 3.10; the pyatspi-dependent osworld Flask
# server also uses 3.10. (No application packages in the system 3.12 site-packages.)

# ---------------------------------------------------------------------------
# System packages — apps + tooling + media-transform CLIs
# ---------------------------------------------------------------------------
#
# Pins captured 2026-05-13 from the live `cua-lite/lite.osworld:latest` image
# (dpkg -l survey). Bump pins deliberately; never floating.
#
# Why each lives here (not base):
#   - LibreOffice / GIMP / VLC / Thunderbird / Chrome / VS Code: task apps,
#     bump independently of GNOME stack
#   - evince:        PDF viewer (xdg-open on .pdf needs it for /setup/open_file wmctrl)
#   - libreoffice-gtk3: AT-SPI accessibility bridge for LibreOffice
#   - python3-pyatspi: AT-SPI Python bindings for the eval probes (at-spi2-core
#     itself now comes from the shared sandbox base — not re-installed here)
#   - ffmpeg already in base; reused by exec-stdio server.py for x11grab screenshots
#   - pandoc / pdftk-java / jq: multi_apps config helpers → RELOCATED to /opt/env/bin
#     (env-facing, off the agent PATH). expect and poppler-utils stay in /usr/bin
#     (guest has them).
#     ImageMagick + xdotool are the BASE's job now (relocated to /opt/env/bin there);
#     osworld neither installs nor relocates them.
#   - mousepad / curl / nano / vim / cron / expect: shell helpers
#
# Provided by the base image instead of this Dockerfile:
#   - gnome-shell-common / gnome-settings-daemon-common stub schemas →
#     base now ships the REAL gnome-shell + gnome-settings-daemon, so the
#     org.gnome.shell.* / org.gnome.settings-daemon.* schemas come bundled
#   - gnome-terminal / nautilus → base already has them
#   - dconf-cli → base ships dconf-{cli,service,gsettings-backend}
#   - wmctrl → base
# Most packages pinned for reproducibility. The unpinned ones below were
# pinned originally but the upstream mirror dropped those versions over time:
#   - libreoffice-* : 1:7.3.7-0ubuntu0.22.04.10 — removed when the .11/.12
#                    point releases shipped. We accept whatever's current in
#                    jammy-updates; OSWorld doesn't pin to libreoffice
#                    internals so any 7.x is fine.
#   - google-chrome-stable (further below): Google's apt/direct download only
#                    keeps current stable reliably. The downstream `apt-mark
#                    hold` freezes the version that lands here for the lifetime
#                    of the image.
# If more pins start failing the same way, either unpin selectively here or
# switch the whole RUN to a snapshot.ubuntu.com mirror (see docs).
RUN DEBIAN_FRONTEND=noninteractive apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libreoffice-writer \
        libreoffice-calc \
        libreoffice-impress \
        libreoffice-math \
        libreoffice-style-yaru \
        libreoffice-gtk3 \
        libreoffice-core \
        libreoffice-common \
        gimp=2.10.30-1ubuntu0.1 \
        gimp-data=2.10.30-1ubuntu0.1 \
        vlc=3.0.16-1build7 \
        vlc-bin=3.0.16-1build7 \
        vlc-data=3.0.16-1build7 \
        vlc-l10n=3.0.16-1build7 \
        thunderbird=1:140.7.1+build1-0ubuntu0.22.04.1 \
        evince=42.3-0ubuntu3.2 \
        evince-common=42.3-0ubuntu3.2 \
        libgdk-pixbuf2.0-bin=2.42.8+dfsg-1ubuntu0.5 \
        ffmpegthumbnailer=2.2.2+dfsg-1 \
        totem=42.0-1ubuntu1 \
        `# gstreamer1.0-libav: avdec_h264 / avdec_aac decoders Totem needs to actually PLAY the` \
        `# task .mp4/.webm (plugins-good, pulled in by totem/vlc, has the demuxers but not the` \
        `# libav decoders → a "decoder not installed" modal; repairs the video→Totem handler switch).` \
        gstreamer1.0-libav=1.20.3-0ubuntu1 \
        python3-pyatspi=2.38.2-1 \
        python3-uno \
        expect=5.45.4-2build1 \
        mousepad=0.5.8-1 \
        pandoc=2.9.2.1-3ubuntu2 \
        poppler-utils=22.02.0-2ubuntu0.13 \
        pdftk-java=3.2.2-1 \
        `# full vim (not vim-tiny): the guest agent has full vim, AND the stock vim.desktop's` \
        `# TryExec=vim only resolves with /usr/bin/vim present → fixes the missing dash icon.` \
        `# python3-tk: guest has it (tkinter 3.10.8) — agent clipboard/GUI habits transfer.` \
        `# vim pin bumped .35 -> .36: jammy-updates carries one version at a time, so the` \
        `# older point release stopped resolving and the layer failed to build once a` \
        `# source edit invalidated the cache. Bump deliberately, as the pin policy says.` \
        curl nano vim=2:8.2.3995-1ubuntu2.36 jq cron xclip locales python3-tk \
    && rm -rf /var/lib/apt/lists/*

# Agent/env CLI separation — osworld's parity policy: the guest agent LACKS these CLIs, so move
# them off the agent PATH into /opt/env/bin. Same three-step relocate → payload → remove ordering
# the base uses (Dockerfile.linux). Step 1: the SHARED helper (baked at /opt/lite/relocate-clis.sh
# by the base) relocates ONLY — never removes. `java` folds into its `*)` branch (real binary is
# the JRE under /usr/lib/jvm, so symlinked env-side, not moved). xdotool + ImageMagick are the
# BASE's job now (already vendored to /opt/env/bin by the base + their /usr/bin install dropped),
# so osworld neither installs nor relocates them. `expect`/`xsel` are deliberately NOT vendored:
# `expect` stays on /usr/bin (agent-visible — accepted lite-only deviation); `xsel` is un-baked
# so a clipboard task `apt install`s it on demand (VM-faithful).
# 1. relocate commands (helper, no removal)
RUN sh /opt/lite/relocate-clis.sh xclip jq pandoc pdftk java

# 2. relocate payloads BEFORE any removal, then 3. remove the COMMAND packages LAST + re-seal
# /opt/env (the helper's chmod ran before the payload move, so it does not cover the jar).
# pdftk-java bundles the launcher (relocated above) AND its payload jar in ONE package, so a
# remove-before-relocate would delete /usr/share/pdftk/pdftk.jar → move it first + repoint the
# launcher (the bcprov/commons-lang3 dep-jars are separate packages that survive the remove).
# The JRE sibling CLIs (keytool/rmiregistry/jjs/jexec/...) the pdftk-java JRE registers on
# /usr/bin are agent divergences too (the guest has no java ecosystem). Derive from EVERY
# /usr/bin symlink resolving into /usr/lib/jvm — bin/ AND lib/ (jexec lives in lib/) — so the
# set is complete-by-construction; remove-only (nothing invokes them; skip java, done above).
# The JRE (openjdk-11-jre-headless) is intentionally NOT removed (it backs /opt/env/bin/java +
# LibreOffice) → orphaned, and the dropped autoremove is what keeps /usr/lib/jvm.
RUN mkdir -p /opt/env/share/pdftk \
    && mv /usr/share/pdftk/pdftk.jar /opt/env/share/pdftk/pdftk.jar \
    && sed -i 's#/usr/share/pdftk/pdftk.jar#/opt/env/share/pdftk/pdftk.jar#' /opt/env/bin/pdftk.pdftk-java \
    && for l in /usr/bin/*; do [ -L "$l" ] || continue; \
           case "$(readlink -f "$l" 2>/dev/null)" in \
             /usr/lib/jvm/*) c="$(basename "$l")"; [ "$c" = java ] && continue; \
                             update-alternatives --remove-all "$c" >/dev/null 2>&1 || true; rm -f "$l" ;; \
           esac; \
       done \
    && DEBIAN_FRONTEND=noninteractive apt-get remove -y pdftk-java jq xclip pandoc \
    && chmod -R a+rX /opt/env

# ImageMagick by default refuses PDF / PS / EPS read+write due to a 2018 CVE
# (ImageMagick-PolicyXML). Synth tasks legitimately need PDF read+write
# (rasterize SVG icons, build contact-sheet PDFs, etc.). Relax the policy
# on PDF / PS / EPS only — keep the global policy file intact for everything
# else. The rule lives at /etc/ImageMagick-6/policy.xml on Ubuntu 22.04.
RUN sed -i 's|<policy domain="coder" rights="none" pattern="PDF" />|<policy domain="coder" rights="read\|write" pattern="PDF" />|' /etc/ImageMagick-6/policy.xml

# Re-open the /opt/env wall for the desktop-user server (a+rX: read + traverse
# dirs + exec binaries, no write). Env CLIs stay vendored in /opt/env/bin.
RUN chmod -R a+rX /opt/env

# OSWorld VM uses en_HK.UTF-8 as the system locale (verified via `locale` probe
# in the guest VM). Generate BOTH en_US (some evaluators depend on its ls sort
# order) and en_HK, and default the container to en_HK to match osworld — this
# aligns spreadsheet number/date formatting + spellcheck dictionary selection.
# LC_ALL is a hard override, so BOTH LANG and LC_ALL must be flipped (leaving
# LC_ALL=en_US would nullify LANG).
RUN sed -i '/en_US.UTF-8/s/^# //' /etc/locale.gen && \
    sed -i '/en_HK.UTF-8/s/^# //' /etc/locale.gen && \
    locale-gen
ENV LANG=en_HK.UTF-8 LC_ALL=en_HK.UTF-8

# Timezone — osworld guest starts in Asia/Hong_Kong (HKT +0800); lite defaulted
# to UTC. Keep timezone as real system state, not process-level `ENV TZ`, so
# later `timedatectl set-timezone` / `/etc/localtime` mutations are observable
# to ordinary shell commands and evaluators.
RUN ln -sf /usr/share/zoneinfo/Asia/Hong_Kong /etc/localtime && \
    echo Asia/Hong_Kong > /etc/timezone

# The OSWorld VM exposes systemd's `timedatectl`; this container does not run
# systemd, but OSWorld-family tasks and evaluators still expect the VM-shaped
# command output. Install an osworld-scoped shim in the image so setup does not
# rewrite shell scripts on every reset.
COPY --chmod=0755 docker/bin/timedatectl /usr/local/bin/timedatectl

# Docker containers cannot mutate the kernel hostname from inside the guest in
# the same way the OSWorld VM can. A small osworld-scoped shim preserves the
# task/evaluator contract for `hostname` reads and simple hostname changes while
# delegating option forms like `hostname -i/-I` to the real binary.
COPY --chmod=0755 docker/bin/hostname /usr/local/bin/hostname

# The OSWorld VM exposes power profile state through `powerprofilesctl`. The
# container has no real host power daemon, but OSWorld-family tasks/evaluators
# treat the CLI as user-facing OS state. Keep this shim on the agent-visible
# PATH (`/usr/local/bin`), not `/opt/env/bin`, because it is part of the VM
# surface a user may legitimately manipulate.
COPY --chmod=0755 docker/bin/powerprofilesctl /usr/local/bin/powerprofilesctl

# Make the three VM-parity root shims executable and grant NOPASSWD sudo for all
# three in a single osworld-scoped sudoers drop-in.
RUN printf '%s\n' \
      'user ALL=(ALL) NOPASSWD: /usr/local/bin/timedatectl' \
      'user ALL=(ALL) NOPASSWD: /usr/local/bin/hostname' \
      'user ALL=(ALL) NOPASSWD: /usr/local/bin/powerprofilesctl' \
      > /etc/sudoers.d/99-osworld-shims && \
    chmod 0440 /etc/sudoers.d/99-osworld-shims

# The OSWorld VM exposes the legacy accessibility key
# `org.gnome.desktop.a11y.interface large-text`. The container's current GNOME
# schema only has `org.gnome.desktop.interface text-scaling-factor`, while
# OSWorld-family tasks/evaluators may still call the old key. Intercept only
# that legacy key and delegate every other gsettings call to the real binary.
COPY --chmod=0755 docker/bin/gsettings /usr/local/bin/gsettings

# ---------------------------------------------------------------------------
# Fonts — MS Core fonts + Calibri/Cambria equivalents + Noto CJK
# ---------------------------------------------------------------------------
#
# Microsoft web fonts (Arial, Times New Roman, Courier New, Georgia, Verdana,
# Trebuchet MS, Impact, etc.) — used directly; head-to-head confirmed Arial/Times
# render identically to osworld. Calibri/Cambria are deliberately NOT given the
# metric-compatible Carlito/Caladea: the osworld EVAL VM lacks them and falls back
# to DejaVu (see the fontconfig alias below), so lite drops them too to match the
# eval image's font set (and DejaVu becomes the natural fallback). LibreOffice /
# python-docx tasks reference these names heavily; the strong Calibri→DejaVu /
# Cambria→DejaVu alias below still resolves them to a real installed family, so LO
# emits no "font not available" warning, and docx round-trip preserves the typed
# font.name string regardless (verified via headless soffice round-trip).
RUN echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" \
      | debconf-set-selections && \
    DEBIAN_FRONTEND=noninteractive apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ttf-mscorefonts-installer \
      fonts-noto-cjk fonts-noto-cjk-extra \
      fonts-dejavu-core fonts-noto-core fonts-noto-extra fonts-noto-color-emoji \
      fonts-lato fonts-inter && \
    fc-cache -f && \
    rm -rf /var/lib/apt/lists/*
# Parity: the osworld VM ships a set of Google display/script fonts (Marcellus,
# Oswald, Montserrat, Caveat, Josefin Sans, Gloria Hallelujah, Libre Franklin,
# Playfair Display, Pacifico, Walter Turncoat, ...) that lite lacked → ≥5 impress
# decks rendered a substituted face (title serif→sans, hand-drawn→smooth sans),
# a systematic visual OOD between train (lite) and eval (osworld). fonts-lato /
# fonts-inter come from apt above; the rest are TTF-downloaded just below.
# fonts-dejavu-core + fonts-noto-{core,extra} + color-emoji added after a screenshot
# validation found tofu (□) boxes in agent observations: staged wiki content (mount-everest,
# yoga, coffee, ... — used by chrome HTML tasks AND writer docx via article_key) contains
# Devanagari / Tibetan / Arabic / IPA scripts the prior CJK-only font set could not render;
# and generators reference DejaVu Sans/Serif/Mono (writer source docs, ImageMagick -font,
# PIL/ffmpeg drawtext) which silently substituted to Verdana/TNR without fonts-dejavu.
# core+extra (not the full fonts-noto meta) covers all observed non-CJK scripts without
# duplicating the CJK coverage already provided by fonts-noto-cjk.

# The osworld-only Google display/script faces (see the parity note above), fetched
# as TTFs from the google/fonts repo. Additive/non-critical: a single 404 only WARNs
# (upstream occasionally moves paths) so it never fails the build. Variable fonts carry
# the whole weight axis in one file, so registering the family is enough to stop LO
# substitution. Bracket chars in VF filenames are URL-encoded so curl doesn't glob them.
COPY docker/fonts/google-fonts.txt /tmp/osworld-google-fonts.txt
RUN mkdir -p /usr/share/fonts/truetype/googlefonts && \
    while read -r url dest; do \
        curl -fsSL "$url" -o "/usr/share/fonts/truetype/googlefonts/$dest" || echo "WARN: font $dest failed"; \
      done < /tmp/osworld-google-fonts.txt && \
    rm -f /tmp/osworld-google-fonts.txt && \
    fc-cache -f

# Rollout validation: 43% of OSWorld eval xlsx files default to
# Chinese fonts (DengXian, SimSun, Source Han Sans) baked in by Chinese-version
# Excel. Without those exact font names installed, LO's font substitution emits
# a "Font Name. The current font is not available and will be substituted"
# tooltip that consistently distracts the agent across every font-related
# click. Install Noto CJK (covers SC/TC/JP/KR glyphs) and add fontconfig aliases
# so LO finds a real fallback for the named Chinese fonts used in source xlsx
# files.
COPY docker/fonts/conf.d/49-cjk-aliases.conf /etc/fonts/conf.d/49-cjk-aliases.conf
RUN fc-cache -f

# LibreOffice defaults to the generic X11 VCL plugin which doesn't support
# AT-SPI accessibility.  Force GTK3 so the AT-SPI bridge (gail:atk-bridge)
# is loaded and evaluators that check the accessibility tree work correctly.
ENV SAL_USE_VCLPLUGIN=gtk3

# ---------------------------------------------------------------------------
# Google Chrome (current stable, then pinned via apt-mark hold)
# ---------------------------------------------------------------------------
#
# Build reliability comes first: historical Chrome .deb mirrors expire or
# disappear, while Google's official direct package serves current stable. Hold
# the installed version inside the image so container-side apt operations cannot
# drift it afterward. Override CHROME_DEB_URL only when a durable pinned artifact
# is available.
ARG CHROME_DEB_URL=https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
RUN DEBIAN_FRONTEND=noninteractive apt-get update -qq && \
    curl -fL --retry 3 --connect-timeout 20 "${CHROME_DEB_URL}" -o /tmp/chrome.deb && \
    dpkg-deb -f /tmp/chrome.deb Version && \
    printf 'repo_add_once=false\nrepo_reenable_on_distupgrade=false\n' \
      > /etc/default/google-chrome && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq /tmp/chrome.deb && \
    apt-mark hold google-chrome-stable && \
    rm -f /etc/cron.daily/google-chrome \
          /opt/google/chrome/cron/google-chrome \
          /tmp/chrome.deb && \
    rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# VS Code — pinned + apt-mark hold
# ---------------------------------------------------------------------------
#
# Pinned to 1.95.3, the last stable release BEFORE Microsoft bundled
# GitHub Copilot's "Sign in to continue with AI-powered development"
# onboarding modal (introduced in 1.96, Dec 2024). Newer releases display
# that modal on first launch and it sits on top of the Settings UI, blocking
# all eval task interaction (4 vs_code settings tasks scored
# 0/16 because every variant terminated within ~5 turns trying to dismiss
# the modal). 1.95.3 has all modern features needed for our perturb/eval
# tasks (settings.json, keybindings.json, multi-cursor, file ops) without
# the AI nag. Bump the pinned version deliberately (and re-validate) only
# when the bundled tooling needs to advance.
#
# .deb archive URL: https://update.code.visualstudio.com/<ver>/linux-deb-x64/stable
ARG VSCODE_VERSION=1.95.3
RUN curl -fsSL "https://update.code.visualstudio.com/${VSCODE_VERSION}/linux-deb-x64/stable" \
      -o /tmp/code.deb && \
    DEBIAN_FRONTEND=noninteractive apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq /tmp/code.deb && \
    apt-mark hold code && \
    rm -f /tmp/code.deb && \
    rm -rf /var/lib/apt/lists/*

# VS Code --no-sandbox wrapper (same fix as the google-chrome wrapper below):
# Docker's default seccomp blocks unprivileged user-namespace creation, so
# Chromium/Electron's zygote sandbox FATALs and `code <path>` opens no window.
# Shadow the binary on PATH (/usr/local/bin ahead of /usr/bin) + repoint the
# dock/app-grid launcher so every cold start passes --no-sandbox. Only Chromium/
# Electron apps need this (find / -name chrome-sandbox → chrome + code); GTK/Qt
# apps and Gecko (Thunderbird, degrades gracefully) are unaffected.
COPY --chmod=0755 docker/software/vscode/bin/code /usr/local/bin/code
RUN rm -f /usr/bin/code && \
    install -m 0755 /usr/local/bin/code /usr/bin/code
RUN sed -i 's#Exec=/usr/share/code/code#Exec=/usr/bin/code#' \
    /usr/share/applications/code.desktop

# ---------------------------------------------------------------------------
# Python packages — TWO distinct concerns, tagged + split below:
#   (A) [ISOLATION]           the ENV venv (/opt/env/venv) EXTEND + re-open a+rX
#                             — a boundary step; the FIRST RUN below this banner.
#   (B) [AGENT-PYTHON parity] the SYSTEM-3.10 guest mirror — agent OOD-parity, NOT
#                             an isolation step; the SECOND RUN, further below.
# ---------------------------------------------------------------------------
#
# AGENT / ENV PACKAGE SEPARATION — mirrors osworld's guest-VM(agent) vs host(eval)
# split so a policy trained on lite transfers to osworld.
#
# osworld's AGENT python (guest, 3.10) carries Pillow/numpy/scipy/matplotlib/sympy/
# reportlab/olefile/paramiko/pexpect/pynput/lxml/bs4/requests but NOT the office/
# setup libs (openpyxl/docx/pptx/fitz/mutagen/xlsxwriter/pandas) — osworld runs
# those HOST-side. lite runs agent+config+eval in ONE container, so we replicate:
#   (1) the AGENT python = the SYSTEM 3.10 (below) carrying exactly osworld's agent
#       set — so learned `python3 import X` habits transfer to osworld eval.
#   (2) ALL config/setup/gold/eval python packages live in a SELF-CONTAINED venv
#       /opt/env/venv, reached because the osworld Flask server + exec_stdio carry
#       /opt/env/venv/bin on their PATH (no dispatch python3-rewrite). This keeps
#       setup-only libs (openpyxl/docx/pptx/fitz/mutagen/xlsxwriter/pandas) entirely
#       OFF the agent's python. Import surface in config/gold (task data): office +
#       PIL(212) + lxml(568) + numpy(13) + bs4 + pandas(2).
# Env python = the BASE's /opt/env/venv (python 3.12, created by sandbox.linux with the
# generic capture deps Pillow + python-xlib). We EXTEND that same venv here with osworld's
# setup/eval deps — we do NOT create our own (the base owns the env python; osworld adds
# on top, per the /opt/env architecture). Reached by the env PATH (exec_stdio server runs
# on this python; 50-osworld-server.conf prepends /opt/env/venv/bin), NOT by any dispatch
# python3-rewrite. Includes pyautogui + python-xlib (302 config pyautogui steps) +
# requests/websocket-client (eval CDP getters) + PyYAML (config). Pillow/python-xlib below
# are idempotent (the base already provides them) — kept explicit so pyautogui can't drag
# in a colliding Xlib fork.
# [ISOLATION] (A) env venv EXTEND — add osworld's setup/eval deps to the BASE-owned
# /opt/env/venv, then re-open a+rX. Boundary step (env-only python; off the agent PATH).
RUN /opt/env/bin/uv pip install --python /opt/env/venv/bin/python --no-cache-dir \
        openpyxl==3.1.5 python-pptx==1.0.2 python-docx==1.1.2 \
        mutagen==1.47.0 pymupdf==1.25.5 XlsxWriter==3.2.0 \
        Pillow numpy lxml beautifulsoup4 pandas \
        pyautogui==0.9.54 python-xlib==0.33 requests==2.34.2 websocket-client==1.9.0 PyYAML \
        pysrt==1.1.2 && \
    chmod -R a+rX /opt/env/venv

# [AGENT-PYTHON parity] (B) SYSTEM-3.10 guest mirror — NOT an isolation step (agent OOD-parity).
# OSWorld Flask server + agent-3.10 guest-mirror deps — Flask, pyautogui, pynput, python-xlib.
# (pygame is NOT a server dep — main.py imports zero pygame; see the guest-mirror note below.)
#
# Xlib pin — DO NOT switch back to the `python3-xlib` fork. `pynput` depends on
# the maintained `python-xlib` (0.33), while `python3-xlib` (an old 0.15 fork)
# installs into the SAME `Xlib/` import dir — so requesting both lets their files
# collide non-deterministically, leaving a mixed 0.15/0.33 tree where
# `xauth.py` references `X.FamilyServerInterpreted` that the stale `X.py` lacks.
# `import pyautogui` then dies and the whole Flask server crash-loops (server
# never binds :5000 → every config/oracle step fails). Pin the single maintained
# `python-xlib==0.33` (which pynput also wants) so there is exactly one Xlib.
#
# NOTE: the OSWorld server imports `pyatspi` (AT-SPI Python bindings)
# which only ships as Debian package `python3-pyatspi` for the system python.
# Ubuntu 22.04's system python is 3.10 (under /usr/bin/python3), and the
# compiled `_gi.cpython-310-*.so` won't load in the 3.12 interpreter (the env venv
# at /opt/env/venv) — circular-import error.
# Solution: install the OSWorld-server-only deps into /usr/bin/python3
# (3.10) and run main.py with that interpreter. The 3.12 install above
# stays for the oracle/eval helpers that don't need pyatspi.
# python3.10-dev provides Python.h needed by evdev's sdist build (transitive
# dep of pynput on Linux). It stays in the final image: the purge/autoremove pass
# was dropped (see Step 3 above) so autoremove can't orphan-sweep the vendored JRE
# under /usr/lib/jvm — the build headers ride along (a few MB, acceptable).
# Left UNPINNED: apt drops old patch versions from the jammy mirror on each point
# release (e.g. .15 → .16), so a hardcoded pin rots and the build breaks. Unpinned
# resolves to the current python3.10 patch (apt keeps the runtime + -dev in lockstep).
RUN DEBIAN_FRONTEND=noninteractive apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3.10-dev \
    `# pyautogui + its call-time deps (pyscreeze, pytweening, pymsgbox) are installed` \
    `# --no-deps: pyscreeze is imported at call time (a top-level import check misses it)` \
    `# and needs only Pillow (installed below); pytweening/pymsgbox are pure-python` \
    `# guest-mirror parity.` \
    `# --no-deps is MANDATORY: a full-deps install pulls mouseinfo (which depends on the` \
    `# python3-Xlib 0.15 fork → clobbers python-xlib 0.33 → the crash-loop above) and` \
    `# mouseinfo also sys.exit()s at import without python3-tk, killing import pyautogui;` \
    `# pygetwindow raises NotImplementedError on Linux at import. So install ONLY the safe` \
    `# subset explicitly, pinned to the osworld guest versions (agent OOD-parity).` \
    && /opt/env/bin/uv pip install --python /usr/bin/python3 --no-cache-dir --no-deps \
        pyautogui==0.9.54 pyscreeze==0.1.30 pytweening==1.0.7 pymsgbox==1.0.9 \
    && /opt/env/bin/uv pip install --python /usr/bin/python3 --no-cache-dir \
        flask==3.0.0 \
        pynput==1.7.6 \
        python-xlib==0.33 \
        lxml==4.8.0 \
        requests==2.25.1 \
        Pillow==10.1.0 \
        `# numpy pinned 1.24.4, NOT the guest's 1.26.2: scipy==1.8.0 metadata caps numpy<1.25, so` \
        `# a clean uv resolution of both together is UNSAT (the guest violated this at install` \
        `# time; we can't). numpy 1.24↔1.26 is agent-immaterial (identical import surface).` \
        numpy==1.24.4 \
        `# agent-3.10 = osworld GUEST mirror: every other version pinned` \
        `# to the guest so a policy's learned "python3 import X" habits transfer 1:1 to osworld eval.` \
        `# ADD the guest-set libs lite's agent lacked; do NOT add office/pandas/websocket-client` \
        `# (guest lacks → they live in the env venv). pygame REMOVED (guest lacks it; the snake` \
        `# eval imports a local pygame.py STUB written to the task cwd by` \
        `# src/gen/eval/multi_apps.py:311-360, NOT system pygame; the server main.py has zero` \
        `# pygame imports → env-safe). html5lib ADDED (guest has 1.1; HTML parsing).` \
        scipy==1.8.0 matplotlib==3.5.1 sympy==1.9 reportlab==3.6.8 olefile==0.46 \
        paramiko==2.9.3 pexpect==4.8.0 beautifulsoup4==4.10.0 pyperclip==1.8.2 html5lib==1.1 \
        `# PyYAML NOT pip-installed: the guest's yaml is apt python3-yaml 5.4.1 (installed in the` \
        `# base), so omitting a pip PyYAML makes the agent's yaml exactly the guest's 5.4.1. A pip` \
        `# PyYAML 6.0.3 would instead shadow it under /usr/local with a guest-mismatched version.` \
        `# mouseinfo/pygetwindow/pyrect are omitted though the guest has them — an accepted parity` \
        `# deviation: installing them pulls python3-xlib 0.15, which clobbers python-xlib 0.33 (the` \
        `# Xlib crash-loop above), and pygetwindow raises NotImplementedError on Linux at import.` \
    # CRITICAL: the `uv pip install --python /usr/bin/python3` calls above install
    # into the SYSTEM python3.10 and clobber the apt-managed setuptools *files*
    # (dpkg still reports python3-pkg-resources installed, but the on-disk
    # /usr/lib/python3/dist-packages/pkg_resources dir is gone). supervisord —
    # the container's PID 1, whose shebang is /usr/bin/python3 (3.10) — imports
    # pkg_resources at startup, so without this the container dies at boot with
    # "ModuleNotFoundError: No module named 'pkg_resources'". Reinstall the files
    # LAST (this is the final step touching 3.10's site-packages). This layer is the
    # ONLY place in the whole family that disturbs 3.10 on purpose (the agent guest
    # mirror): the shared base never runs a system-level pip at all — it builds the
    # env venv with `uv venv` + `uv pip --python` (never a system pip) — so the base needs no such reinstall.
    && DEBIAN_FRONTEND=noninteractive apt-get install --reinstall -y python3-pkg-resources \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# OSWorld Flask server (/setup, /file, /accessibility, /execute) on :5000
# ---------------------------------------------------------------------------
#
# Copy server code into the image, then register a supervisord program.
# The script waits for the X server + the DBUS_SESSION_BUS_ADDRESS file
# (written by the base's start-gnome.sh) before exec'ing main.py.
COPY docker/server/main.py /home/user/server/
RUN chown -R user:user /home/user/server

# OSWorld Flask server startup wrapper.
# Gates on GNOME-Shell readiness (set by /usr/local/bin/wait-for-gnome-ready.sh
# from the shared sandbox base). Without a live session, gsettings/gtk/atk-bridge calls
# from setup steps fail and a11y getters return empty trees — fail-fast so
# port 5000 never opens and downstream env-side errors surface clearly.
# Run with system python 3.10 (has python3-pyatspi + python3-gi
# from apt). The 3.12 env venv (/opt/env/venv) cannot import the
# cpython-310-built _gi.so.
COPY --chmod=0755 docker/server/start-osworld-server.sh /usr/local/bin/start-osworld-server.sh

# Supervisor program for OSWorld Flask server (port 5000)
COPY docker/supervisor/50-osworld-server.conf /etc/supervisor/conf.d/50-osworld-server.conf

# ---------------------------------------------------------------------------
# App pre-warming and configuration
# ---------------------------------------------------------------------------

RUN mkdir -p /home/user/Desktop && chown user:user /home/user/Desktop

# Create empty ~/.Xauthority for user. Xvnc is launched with
# `-SecurityTypes None` in the base, so it never writes one, but the
# OSWorld Flask server's pyautogui (via python3-xlib) raises XauthError
# on missing file even when DISPLAY connects successfully. An empty file
# is sufficient — Xlib falls back to "no xauthority details" and the X
# connect succeeds.
RUN touch /home/user/.Xauthority && chown user:user /home/user/.Xauthority

# Terminal parity with the osworld VM.
# The sandbox base created `user` WITHOUT copying /etc/skel (skel only seeds NEW
# accounts, and the base pre-created the home), so `user` has NO ~/.bashrc. bash
# then falls back to the bare default: a PLAIN white `user@host:~$` prompt and a
# static "Terminal" titlebar. The osworld VM's `user` carries the stock Ubuntu
# ~/.bashrc, which (a) colors the prompt (green user@host, blue cwd — enabled when
# TERM is *-256color, as gnome-terminal sets) and (b) sets the window title to
# `user@host: cwd` via the `\e]0;...\a` PROMPT escape. Seed the SAME stock skel
# files (unchanged — faithful to the VM, not "nicer") so the container terminal
# renders identically. `-n` never clobbers a file a later step may install.
RUN for f in .bashrc .profile .bash_logout; do \
        [ -f /etc/skel/$f ] && cp -n /etc/skel/$f /home/user/$f || true; \
    done && chown user:user /home/user/.bashrc /home/user/.profile /home/user/.bash_logout 2>/dev/null || true

# LibreOffice pre-warm — populates the profile dir + caches font metrics.
RUN echo "test" > /tmp/warmup.txt && \
    sudo -u user libreoffice --headless --convert-to odt /tmp/warmup.txt --outdir /tmp/ 2>/dev/null; \
    rm -f /tmp/warmup.txt /tmp/warmup.odt

# LibreOffice disable popups + suppress first-run banner.
# The previous /etc/skel/.config/libreoffice/... block was a
# no-op because the `user` account already exists in the base image —
# /etc/skel only seeds new accounts. Consolidated the FirstStartWizardCompleted
# / FirstRun keys into the live /home/user profile here. Banner is critical:
# "You are running version 7.3 of LibreOffice for the first time. Do you want
# to learn what's new?" pushes UI elements down ~30px and throws off agent
# coordinate models (observed: agents miss the cell name box at (76,171) by
# clicking (76,168) → font name dropdown).
# WarnAlienFormat=false also pairs with the LO_SAVE_POSTCONFIG fix in
# common.py so ctrl+s on .xlsx/.docx/.pptx doesn't trigger the
# "Use Excel Format!" / "Keep Current Format" dialog.
# Writer Release Notes infobar validation confirmed 9+ writer tasks fail
# because the LO 7.3 Release Notes
# infobar at y≈155-190 still appears (despite FirstStartWizard / FirstRun
# suppression) and intercepts agent toolbar clicks at y∈[170,180]. Added
# ooSetupShowReleaseNotes=false + HelpUserAssistance/HelpAgentEnabled=false
# below to cover the remaining first-run notification paths.
# registrymodifications.xcu is the single canonical LibreOffice config source.
# Bake it as a read-only source under /usr/local/share/osworld/libreoffice; the
# runtime script docker/runtime/apply-osworld-settings.sh cp's it into the live
# user profile on each reset (eval wipes the profile, so a COPY straight into the
# profile here would be dead-weight).
COPY docker/software/libreoffice/registrymodifications.xcu /usr/local/share/osworld/libreoffice/registrymodifications.xcu
RUN mkdir -p /home/user/.config/libreoffice/4/user && \
    chown -R user:user /home/user/.config/libreoffice

# VS Code first-run suppression.
# NOTE: the "Welcome" tab pinned by VS Code's first-run flow lands on top of the
# Settings UI and consumes most of the agent's turns clicking past it, so
# all 4 uniform-zero settings-UI tasks fail uniformly. Pre-seed the user
# settings.json to suppress the welcome page (`workbench.startupEditor` +
# `workbench.welcomePage.walkthroughs.openOnInstall`) and other first-run
# noise that steals agent turns: telemetry, auto-update checks, the chat
# command center, and workspace-trust prompts. Eval tasks that care about
# specific settings will overwrite this file in their setup phase before
# pkill+relaunch.
COPY docker/software/vscode/settings.json /home/user/.config/Code/User/settings.json
RUN mkdir -p /home/user/.config/Code/User && \
    chown -R user:user /home/user/.config/Code

# Thunderbird profile dir.
RUN mkdir -p /home/user/.thunderbird && chown user:user /home/user/.thunderbird
# Suppress Thunderbird "What's New" tab on first launch (version mismatch with
# OSWorld tarball profiles triggers the post-update page).
COPY docker/software/thunderbird/policies.json /usr/lib/thunderbird/distribution/policies.json

# VLC config dir.
RUN mkdir -p /home/user/.config/vlc && chown user:user /home/user/.config/vlc

# Default-handler map (single authoritative mimeapps.list): clear the
# http/https/mailto/news scheme handlers so a stray xdg-open on a URL/mail target
# does NOT auto-launch Chrome/Thunderbird as a side effect (osworld tasks launch
# browsers explicitly). text/plain -> gedit and video/* -> Totem to match osworld
# (osworld opens text in gedit and video in GNOME Videos; lite defaulted text to
# Mousepad and video to VLC — different app chrome the agent observes, plus VLC's
# first-run/no-audio modals). Was in the retired base; osworld-specific.
COPY docker/software/mime/mimeapps.list /home/user/.config/mimeapps.list
RUN chown user:user /home/user/.config/mimeapps.list

# Chrome data dir: OSWorld tasks reference /home/user/.config/google-chrome (the default
# profile path), but we launch Chrome with --user-data-dir=/home/user/chrome-data for
# CDP remote debugging (Chrome 146+).  Symlink so both paths resolve to the same dir.
RUN mkdir -p /home/user/chrome-data/Default && \
    ln -sfn /home/user/chrome-data /home/user/.config/google-chrome && \
    chown -h user:user /home/user/.config/google-chrome && \
    chown -R user:user /home/user/chrome-data && \
    # Suppress Chrome first-run welcome dialog
    touch /home/user/chrome-data/First\ Run && \
    # Web-content locale parity: the osworld reference runs the desktop in en_HK
    # (LibreOffice/dates are British/HK — verified) but its Chrome sends
    # Accept-Language en-US, so sites render US dates/terms ("Jul 13", "Car
    # Rental", USD, US flag). Our en_HK system made lite's Chrome send en-GB
    # ("13 Jul", "Car Hire", UK flag) — a train/eval divergence on any locale-
    # sensitive site. Seed intl.accept_languages=en-US in the profile (Chrome
    # merges its defaults but keeps this user pref) so lite's web content matches
    # osworld WITHOUT changing the en_HK system locale or the Chrome UI language.
    printf '{"intl":{"accept_languages":"en-US,en"}}' > /home/user/chrome-data/Default/Preferences && \
    chown user:user /home/user/chrome-data/First\ Run /home/user/chrome-data/Default/Preferences

# Home-dir parity: the shared base scaffolds empty ~/shared + ~/storage; the
# osworld guest has neither, so remove them here (osworld-layer overlay whiteout —
# base/cuaworld keep theirs) to stop them showing as lite-only folders in the
# agent's home. (chrome-data can't be hidden the same way: 200+ task literals
# hardcode /home/user/chrome-data, so it stays as the real profile dir.)
RUN rmdir /home/user/shared /home/user/storage 2>/dev/null || true

# Suppress gnome-keyring "Choose password for new keyring" dialog if a
# keyring daemon were ever spawned (none autostart under our GNOME-Shell
# session, but be defensive — Chrome can pull libsecret which triggers it).
# Three-pronged fix:
# 1. Remove gnome-keyring autostart entries so the daemon doesn't prompt
RUN rm -f /etc/xdg/autostart/gnome-keyring-pkcs11.desktop \
         /etc/xdg/autostart/gnome-keyring-secrets.desktop \
         /etc/xdg/autostart/gnome-keyring-ssh.desktop || true
# 2. Pre-create a default keyring with empty password so if gnome-keyring
#    somehow starts, it finds an existing keyring and doesn't prompt
RUN mkdir -p /home/user/.local/share/keyrings && \
    echo "login" > /home/user/.local/share/keyrings/default && \
    chown -R user:user /home/user/.local/share/keyrings
# 3. Tell Chrome to use plaintext password storage (no keyring integration)
#    via Chrome's Local State config file. Explicit /usr/bin/python3 (3.10) — this
#    only needs the stdlib (json/os), and pinning avoids any PATH ambiguity.
RUN /usr/bin/python3 -c "\
import json, os; \
path = '/home/user/chrome-data/Local State'; \
state = json.load(open(path)) if os.path.exists(path) else {}; \
state['os_crypt'] = {'backend': 'basic'}; \
json.dump(state, open(path, 'w'))" && \
    chown user:user /home/user/chrome-data/Local\ State

# ===========================================================================
# GNOME-Shell-like appearance + runtime settings — moved here from the retired
# lite.osworld.base (osworld-specific vision-baseline parity; the shared base
# ships only a plain GNOME-Shell desktop with a generic dock).
# ===========================================================================

# Deliberately NO Chrome managed-policy file: any file under
# /etc/opt/chrome/policies/managed/ makes Chrome show "Your browser is managed by
# your organization" on every settings page. The keys such a file would set
# (PromotionalTabsEnabled / WelcomePageOnOSUpgradeEnabled / DefaultBrowserSettingEnabled
# = false) are already covered by the wrapper flags below (--no-first-run
# --no-default-browser-check --disable-features=ChromeWhatsNewUI). The
# "Can't update Chrome / Reinstall" nag (osworld parity #8) is ALSO handled by a
# wrapper flag, not a policy: old pinned builds can trip UpgradeDetector against
# the host clock even fully offline, and --simulate-outdated-no-au="1 Jan 2099"
# schedules the simulated-outdated state for a date that never arrives. Keep the
# wrapper flag with current stable too; it is harmless and avoids a future nag if
# the image ages in place.

# google-chrome ANGLE/SwiftShader wrapper: under Xvnc there is no GPU, so
# canvas/WebGL pages render blank without these flags. Shadows the real binary
# on PATH (/usr/local/bin ahead of /usr/bin). Also carries --no-sandbox --test-type
# so the GUI/dock launch path suppresses the yellow "unsupported flag --no-sandbox"
# infobar (it otherwise shifts page content down and offsets click coordinates).
# --password-store=basic forces the plaintext credential backend so a fresh
# terminal-launched Chrome never activates Secret-Service D-Bus → no "Choose a
# password for the new keyring" modal (36037439/48c46dc7). CDP/cookies unaffected.
COPY --chmod=0755 docker/software/chrome/bin/google-chrome /usr/local/bin/google-chrome
# Repoint the shipped launcher (used by the dock + app-grid) at the wrapper so the
# dock/GUI launch path also gets the ANGLE + --test-type flags.
RUN sed -i 's#Exec=/usr/bin/google-chrome-stable#Exec=/usr/local/bin/google-chrome#' \
    /usr/share/applications/google-chrome.desktop

# Dock parity: the real osworld VM pins "Ubuntu Software" (orange bag) + "Help" (blue
# ?) after the app set. Install their launchers so lite's dash renders the same two
# trailing icons — no osworld task uses either (grep-confirmed), but a vision agent
# trained on lite must see the SAME dock as the osworld eval, else the dock's icon
# count/positions drift (OOD). yelp = GNOME Help (org.gnome.Yelp / yelp.desktop);
# gnome-software (deb) = org.gnome.Software.desktop, themed by Yaru as the orange
# Ubuntu-Software bag. Delete gnome-software's background autostart service so it never
# runs unprompted — an idle update daemon would raise its own "updates available"
# banner, which is exactly the kind of lite-only OOD notification we're removing.
RUN DEBIAN_FRONTEND=noninteractive apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        yelp \
        gnome-software \
    && rm -f /etc/xdg/autostart/gnome-software-service.desktop \
    # gnome-software ships Icon=org.gnome.Software (Yaru renders it as a WHITE bag) +
    # Name=Software, but the osworld VM's dock shows the ORANGE Ubuntu-Software tote
    # bag labelled "Ubuntu Software" (the Yaru `ubuntusoftware` icon = software-store).
    # Point the launcher at that icon/name so lite's dock matches osworld pixel-for-icon.
    && sed -i -e 's/^Icon=org\.gnome\.Software$/Icon=ubuntusoftware/' \
              -e 's/^Name=.*/Name=Ubuntu Software/' \
        /usr/share/applications/org.gnome.Software.desktop \
    # The osworld VM's favorite-apps entry is `snap-store_ubuntu-software.desktop`
    # (snap-store isn't installable here — needs snapd — so we ship a stub .desktop
    # of that exact id that launches the themed gnome-software, giving byte-parity
    # on the dconf favorite-apps string without pulling snapd).
    && cp /usr/share/applications/org.gnome.Software.desktop \
          /usr/share/applications/snap-store_ubuntu-software.desktop \
    && rm -rf /var/lib/apt/lists/*

# Dock content: osworld's app set on GNOME-Shell's ubuntu-dash. Overwrite the base's
# generic favorite-apps default (same dconf file) with the OSWorld-baseline launchers
# (chrome/thunderbird/code/libreoffice/gimp/vlc/files/terminal + Software/Help) a
# vision agent expects. A task that sets org.gnome.shell favorite-apps at runtime still
# overrides this (the user dconf db wins over this system-db default), so per-task docks
# work natively.
COPY docker/desktop/dconf/local.d/03-shell-dock /etc/dconf/db/local.d/03-shell-dock

# File-chooser parity: the osworld VM seeds the GTK file-chooser prefs (show hidden
# files, folders NOT sorted first) so every Open/Save dialog lists files in the same
# order + set as the eval reference; lite set none → a different listing/order the
# agent navigates. Workflow-relevant (file nav), so match osworld's defaults.
COPY docker/desktop/dconf/local.d/04-file-chooser /etc/dconf/db/local.d/04-file-chooser

# Desktop-icon layout parity: the osworld reference anchors the desktop-icon grid
# at the BOTTOM-right and hides the Trash icon; the shared base defaults to
# top-right + Trash shown (02-desktop-icons). Override here (osworld-layer, higher
# priority than 02) so desktop file icons land in the same grid cells as osworld —
# otherwise "file on the desktop" click targets and new-file positions diverge.
COPY docker/desktop/dconf/local.d/05-desktop-icons /etc/dconf/db/local.d/05-desktop-icons

# Head-to-head parity fixes (REVISED FIX LIST — full-336 vs frozen osworld):
#  - Nautilus show-hidden OFF: osworld hides LibreOffice `.~lock.*` files; lite showed them
#    (spurious clickable icons → misclick/OOD). (GTK file-CHOOSER stays show-hidden per 04.)
#  - image thumbnails + Recent: osworld's Files/file-chooser show real previews + a Recent
#    sidebar entry; lite had generic icons + no Recent.
COPY docker/desktop/dconf/local.d/06-osworld-parity /etc/dconf/db/local.d/06-osworld-parity
RUN dconf update

# Thumbnail sandbox bypass (osworld-parity: real file previews). GNOME 42's
# gnome-desktop thumbnail factory (Nautilus / desktop-icons / GTK file-chooser)
# wraps EVERY thumbnailer in `bwrap --unshare-all`; Docker's default
# seccomp/apparmor blocks unprivileged user-namespace creation (the
# DockerProvisioner adds no --security-opt), so bwrap aborts ("No permissions to
# create new namespace") and every preview fails -> generic mime icons instead of
# real image/PDF/video thumbnails (osworld shows previews; e.g. PDFs show the
# evince page-preview, not Yaru's own red mime icon). gnome-desktop 42 hardcodes
# bwrap with no unsandboxed fallback and no bypass env var, and bubblewrap is a
# hard dep of nautilus/xdg-desktop-portal (can't remove). Replace bwrap with a
# passthrough shim that drops the namespace flags, applies the --bind path remaps
# to the wrapped command's args (so the factory's tmp output path resolves), sets
# --setenv/--chdir, and execs the thumbnailer directly. Safe: the container is
# already the isolation boundary. (libgdk-pixbuf2.0-bin + ffmpegthumbnailer,
# installed above, supply the missing raster-image + video thumbnailers;
# show-image-thumbnails='always' is set in 06-osworld-parity.)
RUN cp -a /usr/bin/bwrap /usr/bin/bwrap.real
COPY --chmod=0755 docker/desktop/bin/bwrap /usr/bin/bwrap

# XDG file-nav bookmark parity: osworld's Nautilus/GTK file-chooser sidebar lists Desktop
# (+ Recent, enabled above); lite had neither, so file-nav to a Desktop target diverged.
# Seed the Desktop bookmark for the desktop user (~/Desktop already exists).
COPY docker/desktop/gtk/bookmarks /home/user/.config/gtk-3.0/bookmarks
RUN mkdir -p /home/user/.config/gtk-3.0 \
    && chown -R user:user /home/user/.config/gtk-3.0
# (text/plain -> gedit and video/* -> Totem defaults are set in the single
# authoritative mimeapps.list write earlier — search "Default-handler map".)

# NOTE: the generic GNOME-Shell appearance (Ubuntu session + ubuntu-dock, Yaru theme
# + clock + a11y + wallpaper via /usr/local/bin/apply-settings.sh at supervisor
# priority 35) now lives in the SHARED base (lite/gym/sandbox/docker/Dockerfile.linux)
# so every sandbox env looks the same. Only the osworld app-SPECIFIC runtime settings
# stay here, at priority 36 (after the base's 35): VS Code singleton-lock cleanup + the
# LibreOffice v7.3 first-run banner suppression (eval's xdg-open re-inits the LO
# profile at runtime, wiping the build-time preseed — so re-seed on every boot).
COPY --chmod=0755 docker/runtime/apply-osworld-settings.sh /usr/local/bin/apply-osworld-settings.sh
COPY docker/supervisor/36-apply-osworld-settings.conf /etc/supervisor/conf.d/36-apply-osworld-settings.conf

# NOTE: the exec-stdio in-container server (/opt/lite/stdio_server.py) is baked by
# the SHARED base (Dockerfile.linux COPYs exec_stdio/server.py there). osworld's
# freshness hash includes that source, so a server.py change forces a rebuild of
# both — no need to re-bake it here (was a redundant COPY).

# Agent `python`/`python3` = the jammy 3.10 (osworld guest mirror): build-time AND
# runtime `python3` = /usr/bin/python3 = 3.10 (WITHOUT the env office/pandas/websocket
# libs — those live in /opt/env/venv, reached only via the env PATH). Bare `python` →
# `python3` is now registered ONCE in the SHARED base (Dockerfile.linux, via
# update-alternatives → /usr/bin/python3, version-agnostic): base's system python is
# 3.10, so bare `python` lands on the 3.10 agent set and the OSWorld guest parity
# (guest `python` == `python3` == 3.10) holds by inheritance — no per-env alias here.

# Keep the apt package indices in the FINAL image (do NOT leave /var/lib/apt/lists empty).
# The osworld VM guest ships populated indices; a few clipboard tasks do a bare
# `apt install xsel` at setup (no self-`apt-get update`), so baking the indices lets that
# resolve from cache with NO per-boot `apt-get update` — the reset critical path stays free
# of network I/O (north-star D). ~40-60 MB; VM-faithful (the guest has these lists too).
RUN DEBIAN_FRONTEND=noninteractive apt-get update -qq

# Expose OSWorld server port (noVNC :6901 already exposed by base)
EXPOSE 5000
