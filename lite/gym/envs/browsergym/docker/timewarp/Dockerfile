FROM python:3.11-slim-trixie

# cua-lite/browsergym.timewarp: TimeWarp's three self-hosted sites (Wiki, News,
# Shop) for one UI version, selected at `docker run` time with TIMEWARP_VERSION=1..6.
# The browser stays on the host (BrowserGym drives it there); this image only
# serves the sites. Upstream source and site data are pinned below.

ARG TIMEWARP_REPO=https://github.com/sparklabutah/timewarp.git
ARG TIMEWARP_SHA=e3b2b6db636bf83eee3be438268fe8b4cb5f5b31
ARG TIMEWARP_DATA_REPO=https://huggingface.co/datasets/sparklabutah/timewarp-env-data
ARG TIMEWARP_DATA_REV=bfed490a8b4b5dae94b126be6105b84445dcbc4b
ARG SPACY_MODEL_URL=https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TIMEWARP_HOME=/opt/timewarp \
    JAVA_HOME=/usr/lib/jvm/timewarp-java

# pyserini 1.3 bundles an Anserini fatjar compiled for Java 21.
RUN apt-get update -qq \
 && apt-get install -y -qq --no-install-recommends \
      ca-certificates \
      curl \
      git \
      patch \
      procps \
      openjdk-21-jre-headless \
 && rm -rf /var/lib/apt/lists/* \
 && ln -s "/usr/lib/jvm/java-21-openjdk-$(dpkg --print-architecture)" "${JAVA_HOME}"

RUN git init -q "${TIMEWARP_HOME}" \
 && cd "${TIMEWARP_HOME}" \
 && git remote add origin "${TIMEWARP_REPO}" \
 && git fetch -q --depth 1 origin "${TIMEWARP_SHA}" \
 && git checkout -q FETCH_HEAD \
 && rm -rf .git

# TimeWarp's own conda env (environment.yml) pins every package version; reuse
# those pins as pip constraints so the sites run on the versions they ship with.
# torch comes from the CPU index first so pyserini does not pull a CUDA build.
COPY requirements.txt /tmp/timewarp-requirements.txt
RUN sed -n -E 's/^ +- +([A-Za-z0-9_.-]+==[^ ]+)$/\1/p' "${TIMEWARP_HOME}/environment.yml" \
      > /tmp/timewarp-constraints.txt \
 && grep -q '^pyserini==' /tmp/timewarp-constraints.txt \
 && pip install --no-cache-dir -c /tmp/timewarp-constraints.txt \
      --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install --no-cache-dir -c /tmp/timewarp-constraints.txt \
      -r /tmp/timewarp-requirements.txt "${SPACY_MODEL_URL}" \
 && rm /tmp/timewarp-requirements.txt /tmp/timewarp-constraints.txt

# cua-lite deltas to upstream, one reviewable file each (see patches/):
#   bind-all-interfaces.patch        Wiki/News listen on 0.0.0.0 so the
#                                    published container ports are reachable.
#   webshop-warm-before-listen.patch Shop builds its search engine before its
#                                    port opens, so readiness probes cannot race
#                                    the lazy first-request initialization.
COPY patches/ /tmp/timewarp-patches/
RUN cd "${TIMEWARP_HOME}" \
 && for p in /tmp/timewarp-patches/*.patch; do patch -p1 < "$p"; done \
 && rm -rf /tmp/timewarp-patches

# Site data, mirrored on Hugging Face at a pinned revision. The Shop reads only
# the *_1000 product files (upstream setup.sh `-d small`).
RUN data="${TIMEWARP_DATA_REPO}/resolve/${TIMEWARP_DATA_REV}" \
 && curl -fsSL -o "${TIMEWARP_HOME}/env/wiki/wiki_index.pkl" "${data}/wiki_index.pkl" \
 && curl -fsSL -o "${TIMEWARP_HOME}/env/news/news_index.pkl" "${data}/news_index.pkl" \
 && mkdir -p "${TIMEWARP_HOME}/env/webshop/data" \
 && for f in items_shuffle_1000.json items_ins_v2_1000.json items_human_ins.json; do \
      curl -fsSL -o "${TIMEWARP_HOME}/env/webshop/data/${f}" "${data}/webshop/${f}"; \
    done

# Build the Shop's Lucene index exactly as upstream setup.sh does.
RUN cd "${TIMEWARP_HOME}/env/webshop/search_engine" \
 && mkdir -p resources resources_100 resources_1k resources_100k indexes \
 && python convert_product_file_format.py \
 && bash run_indexing.sh

COPY entrypoint.sh healthcheck.py /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh

# Wiki, News, Shop.
EXPOSE 5000 5001 5002

HEALTHCHECK --interval=10s --timeout=5s --start-period=180s \
  CMD ["python", "/usr/local/bin/healthcheck.py"]

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
