# One image serves the API and the built React app on port 8000.

FROM node:22-alpine AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CLAIMDESK_STATIC_DIR=/app/web
WORKDIR /app
COPY backend/pyproject.toml ./backend/
COPY backend/claimdesk ./backend/claimdesk
RUN pip install "./backend[azure]" && useradd --create-home --uid 10001 claimdesk
COPY --from=web /web/dist /app/web
USER claimdesk
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')"
CMD ["uvicorn", "claimdesk.api:app", "--host", "0.0.0.0", "--port", "8000"]
