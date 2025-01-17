# start with an image smaller than nvidia's
FROM ghcr.io/coreweave/ml-containers/torch:es-actions-8e29075-base-cuda12.6.3-ubuntu22.04-torch2.5.1-vision0.20.0-audio2.5.0

# Install system dependencies with apt cache
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
ENV UV_LINK_MODE=copy

WORKDIR /app

# install dependencies with uv cache
COPY uv.lock pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked uv sync

# download t5 to hf cache
RUN --mount=type=secret,id=hf_token,required=true \
    --mount=type=cache,target=/root/.cache/huggingface,sharing=locked \
    /app/.venv/bin/huggingface-cli download --token "$(cat /run/secrets/hf_token)" \
    --local-dir /app/black-forest-labs/FLUX.1-dev \
    --include tokenizer_2/ text_encoder_2/ -- black-forest-labs/FLUX.1-dev

COPY model.py sharded_dataset.py train.py utils.py test.sh ./

CMD ["uv", "run", "/app/test.sh"]
