# Use a lightweight Python 3.12 base image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /telco-churn

# Install basic OS dependencies (gcc is often required by ML/Math libraries)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry

# COPY dependency configuration files FIRST (To leverage Docker build caching)
COPY pyproject.toml poetry.lock* /telco-churn/

# Disable Poetry virtualenv creation & install libraries directly into the container
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root

# COPY the remaining application source code (including src and app folders)
COPY . /telco-churn

# (The execution command is omitted here as it will be fully controlled by docker-compose.yml)