FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
ENV AG1_REPO=/opt/alphageometry
ENV MELIAD_ROOT=/opt/meliad/meliad
ENV AG1_CKPT_DIR=/opt/ag_ckpt_vocab_hf
ENV AG1_HF_REPO=abrahamabelboodala/ALPHAGEOMETRY_ag_ckpt_vocab

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    build-essential \
    python3.10 \
    python3.10-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && ln -sf /usr/bin/pip3 /usr/local/bin/pip \
    && python -m pip install --upgrade pip setuptools wheel

ARG AG1_COMMIT=6777cb586cbb46beed28db12dc72c69770b68337
ARG MELIAD_COMMIT=e8af0543441222c1c4c60d58803511f7cf92908b

RUN git clone https://github.com/google-deepmind/alphageometry.git /opt/alphageometry \
    && cd /opt/alphageometry \
    && git checkout ${AG1_COMMIT}

RUN git clone https://github.com/google-research/meliad.git /opt/meliad \
    && cd /opt/meliad \
    && git checkout ${MELIAD_COMMIT}

ARG WORKER_DIR=experiments/geometry_solvers/ag1_runpod_serverless
COPY . /tmp/ag1_worker_context
RUN set -eux; \
    if [ -f /tmp/ag1_worker_context/requirements.txt ]; then \
      SRC=/tmp/ag1_worker_context; \
    elif [ -f /tmp/ag1_worker_context/${WORKER_DIR}/requirements.txt ]; then \
      SRC=/tmp/ag1_worker_context/${WORKER_DIR}; \
    else \
      echo "Could not locate AG1 worker files in build context" >&2; \
      find /tmp/ag1_worker_context -maxdepth 4 -type f | sort | head -200 >&2; \
      exit 1; \
    fi; \
    cp ${SRC}/patches/patch_ag1_for_jax0418.py /tmp/patch_ag1_for_jax0418.py; \
    cp ${SRC}/requirements.txt /tmp/ag1_worker_requirements.txt; \
    mkdir -p /workspace/src; \
    cp -R ${SRC}/src/. /workspace/src/

RUN python /tmp/patch_ag1_for_jax0418.py /opt/alphageometry

RUN python -m pip install "jax[cuda11_pip]==0.4.18" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

RUN python -m pip install -r /tmp/ag1_worker_requirements.txt

# Bake the checkpoint for the first remote-build path. In production, mount a
# RunPod network volume and set AG1_CKPT_DIR to that cache path instead.
RUN python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id=os.environ.get("AG1_HF_REPO", "abrahamabelboodala/ALPHAGEOMETRY_ag_ckpt_vocab"),
    local_dir=os.environ.get("AG1_CKPT_DIR", "/opt/ag_ckpt_vocab_hf"),
    local_dir_use_symlinks=False,
)
PY
CMD ["python", "-u", "/workspace/src/handler.py"]
