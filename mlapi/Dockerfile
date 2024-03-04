## 1. Build venv stage
FROM python:3.10-slim-buster AS venv

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y \
         curl \
         build-essential \
         libffi-dev \
    && rm -rf /var/lib/apt/lists/*

ENV POETRY_VERSION=1.2.1
RUN curl -sSL https://install.python-poetry.org | python -
ENV PATH /root/.local/bin:$PATH

# Install dependencies
WORKDIR /app
RUN python -m venv /app/venv
COPY pyproject.toml poetry.lock ./
RUN . /app/venv/bin/activate \
    && poetry install --only main

########################
## 2. Build prod stage
FROM python:3.10-slim-buster AS prod

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y \
     curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=venv /app/venv /app/venv/
ENV PATH /app/venv/bin:$PATH

WORKDIR /app
COPY ./src ./src
COPY ./morgan-embed-bio-clinical-bert-ddi ./morgan-embed-bio-clinical-bert-ddi

# Health Check
HEALTHCHECK --start-period=30s --interval=15m --timeout=5m \
    CMD curl -f http://locahost:8000/health || exit 1

# run the app
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]