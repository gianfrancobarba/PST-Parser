# Two independent images are built from this file.
#
#   --target train     GPU image carrying the full fine-tuning stack.
#   --target eval      CPU image carrying only what scoring needs.
#
# Scoring imports neither torch nor transformers, which is what lets the second
# image stay small and run anywhere. Keeping the two apart also means a broken
# training environment cannot invalidate published metrics.

# ---------------------------------------------------------------------------
# Shared build stage: resolve the locked environment with the uv cache mounted.
# ---------------------------------------------------------------------------
FROM ghcr.io/astral-sh/uv:0.11.15@sha256:e590846f4776907b254ac0f44b5b380347af5d90d668138ca7938d1b0c2f98d3 AS uv

# ---------------------------------------------------------------------------
# Training image.
#
# The base carries no CUDA of its own: the torch wheels published for cu128
# already bundle the runtime libraries, and the driver comes from the host
# through the NVIDIA Container Toolkit. Starting from an image that also ships
# CUDA would install the whole stack a second time.
# ---------------------------------------------------------------------------
FROM python:3.11-slim-bookworm@sha256:b18992999dbe963a45a8a4da40ac2b1975be1a776d939d098c647482bcad5cba AS train

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_FROZEN=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    HF_HOME=/cache/huggingface

# git lets the run manifest record the commit the image was built from.
#
# The compiler is needed at run time, not at build time: the optimised backend
# compiles its kernels on first use. It links against the CUDA driver library
# that the container runtime injects, so the C library development files must be
# present as well; a compiler on its own cannot produce a shared object.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /usr/local/bin/

WORKDIR /workspace

# Dependencies resolve from the lockfile alone, so this layer is rebuilt only
# when the lockfile changes.
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project --no-dev --extra cu128 --extra train --extra unsloth

COPY src/ src/
COPY configs/ configs/
COPY prompts/ prompts/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --extra cu128 --extra train --extra unsloth

# Only the directories written at run time change ownership. Recursively
# chowning the virtual environment would rewrite every file into a second layer
# and double the size of the image; nothing writes there once it is built.
RUN mkdir -p /cache/huggingface && useradd --create-home --uid 1000 runner \
    && chown -R runner:runner /workspace /cache
USER runner

ENTRYPOINT ["pstparser"]
CMD ["--help"]

# ---------------------------------------------------------------------------
# Evaluation image: reads a prediction file and computes metrics.
# ---------------------------------------------------------------------------
FROM python:3.11-slim-bookworm@sha256:b18992999dbe963a45a8a4da40ac2b1975be1a776d939d098c647482bcad5cba AS eval

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_FROZEN=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /usr/local/bin/

WORKDIR /workspace

COPY pyproject.toml uv.lock README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project --no-dev

COPY src/ src/
COPY configs/ configs/
COPY prompts/ prompts/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev

RUN useradd --create-home --uid 1000 runner && chown -R runner:runner /workspace
USER runner

ENTRYPOINT ["pstparser"]
CMD ["--help"]
