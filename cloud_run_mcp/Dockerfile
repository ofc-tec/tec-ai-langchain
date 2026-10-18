FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev

COPY server.py ./

ENV PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["uv", "run", "--no-sync", "server.py"]