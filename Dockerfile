FROM python:3.12-slim

LABEL maintainer="structbinary"

ENV PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.7.1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_CREATE=false \
    PATH="/opt/poetry/bin:$PATH"

WORKDIR /app
EXPOSE 8000

# Installing build dependencies, Rust, and Poetry
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl python3-dev libffi-dev libssl-dev gcc build-essential \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && . "$HOME/.cargo/env" \
    && curl -sSL https://install.python-poetry.org | python3 - \
    && ln -s /opt/poetry/bin/poetry /usr/local/bin/poetry \
    && rm -rf /var/lib/apt/lists/*

COPY ./pyproject.toml /app/

RUN poetry lock --no-update && \
    poetry install --no-interaction --no-ansi && \
    adduser --disabled-password --no-create-home cloudbrain

COPY ./graph /app/graph
COPY ./server.py /app/
COPY ./ingestion.py /app/
COPY ./agent.py /app/
COPY ./terrform-repository.url /app/

# USER cloudbrain 

CMD ["python", "server.py"]
