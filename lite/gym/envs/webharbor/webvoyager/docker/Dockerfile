FROM python:3.10-bookworm

ARG WEBHARBOR_WEBVOYAGER_REPO=https://github.com/MinorJerry/WebVoyager.git
ARG WEBHARBOR_WEBVOYAGER_SHA=5a7896738c10bfb8b9edccce6bb0e0411f8ae569
ARG WEBHARBOR_REPO=https://github.com/aiming-lab/WebHarbor.git
ARG WEBHARBOR_SHA=af0765d5abd68fb8ec190165d3aa3de333965f84
ARG WEBHARBOR_ASSETS_REVISION=ad6f424f72cada9e6f5c09a58093d0ceeab9c52b

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    WEBHARBOR_WEBVOYAGER_HOME=/opt/webharbor.webvoyager \
    WEBSYN_HOME=/opt/WebSyn \
    WEBHARBOR_HOME=/opt/webharbor \
    CHROME_BIN=/usr/bin/chromium \
    CHROMEDRIVER_BIN=/usr/bin/chromedriver \
    PYTHONPATH=/opt/webharbor.webvoyager:/opt

RUN apt-get update -qq \
 && apt-get install -y -qq --no-install-recommends \
      ca-certificates \
      curl \
      git \
      patch \
      chromium \
      chromium-driver \
      fonts-liberation \
      fonts-noto-cjk \
      fonts-noto-color-emoji \
      procps \
 && rm -rf /var/lib/apt/lists/*

# Chromium is installed unversioned, so the browser is whatever Debian ships on
# build day. Launch it once here: a browser that cannot start otherwise produces
# a healthy-looking image whose every `/reset` fails with an opaque 500 from
# ChromeDriver ("session not created: Chrome instance exited"), because
# `healthcheck.py` only probes the RPC server. Debian's 150.0.7871.46 did exactly
# that -- it traps (SIGTRAP) at launch. Failing the BUILD instead names the bad
# version while there is still something to do about it.
RUN chromium --version \
 && { /usr/bin/chromium --headless=new --no-sandbox --disable-gpu \
        --disable-dev-shm-usage --user-data-dir=/tmp/chromium-probe \
        --dump-dom data:,ok >/dev/null 2>/tmp/chromium-probe.log \
      || { echo "chromium failed to launch -- this image would fail every /reset"; \
           cat /tmp/chromium-probe.log; exit 1; }; } \
 && rm -rf /tmp/chromium-probe /tmp/chromium-probe.log

RUN git init -q /opt/webharbor.webvoyager \
 && cd /opt/webharbor.webvoyager \
 && git remote add origin "${WEBHARBOR_WEBVOYAGER_REPO}" \
 && git fetch -q --depth 1 origin "${WEBHARBOR_WEBVOYAGER_SHA}" \
 && git checkout -q FETCH_HEAD \
 && rm -rf .git

RUN git init -q /opt/webharbor \
 && cd /opt/webharbor \
 && git remote add origin "${WEBHARBOR_REPO}" \
 && git fetch -q --depth 1 origin "${WEBHARBOR_SHA}" \
 && git checkout -q FETCH_HEAD \
 && rm -rf .git

WORKDIR /opt/webharbor.webvoyager

# Upstream requirements are intentionally tiny, but utils.py imports numpy and
# openai==1.1.1 needs an httpx version before the 0.28 client API break.
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir \
      "numpy<2" \
      "httpx<0.28" \
      fastapi \
      "uvicorn[standard]" \
      Flask==3.1.0 \
      Flask-SQLAlchemy==3.1.1 \
      Flask-Login==0.6.3 \
      Flask-WTF==1.2.2 \
      Flask-Bcrypt==1.0.1 \
      Werkzeug==3.1.3 \
      Jinja2==3.1.4 \
      SQLAlchemy==2.0.36 \
      WTForms==3.2.1 \
      email-validator==2.2.0 \
      "huggingface_hub[cli]>=0.25"

# Apply cua-lite deltas to upstream as reviewable patch files (fork-free
# patches/ contract: every change to upstream source is one file in docker/
# patches/, not an inline sed). fetch_assets-no-same-owner.patch: HF asset
# tarballs preserve the uploader's uid/gid; some Docker builders cannot chown
# those high ids while extracting, so unpack assets as the current build user.
# The same patch also makes the default fetch set come from the pinned WebHarbor
# site's directories instead of the mutable dataset's full tarball list.
COPY patches/ /tmp/webharbor.webvoyager-patches/
RUN cd /opt/webharbor \
 && patch -p1 < /tmp/webharbor.webvoyager-patches/fetch_assets-no-same-owner.patch \
 && grep -q -- "--no-same-owner" scripts/fetch_assets.sh \
 && grep -q -- "SITES_TO_FETCH" scripts/fetch_assets.sh \
 && rm -rf /tmp/webharbor.webvoyager-patches \
 && ASSETS_REVISION="${WEBHARBOR_ASSETS_REVISION}" ./scripts/fetch_assets.sh \
 && ./scripts/check_assets.sh \
 && rm -rf sites/.cache \
 && mkdir -p /opt/WebSyn \
 && cp -a sites/. /opt/WebSyn/ \
 && cp websyn_start.sh /opt/websyn_start.sh \
 && cp control_server.py /opt/control_server.py \
 && cp site_runner.py /opt/site_runner.py \
 && chmod +x /opt/websyn_start.sh

RUN mkdir -p /tmp/webharbor.webvoyager-downloads /tmp/webharbor.webvoyager-results /tmp/websyn_pids

COPY server.py entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
