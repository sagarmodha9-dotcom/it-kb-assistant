# Web chat UI. Local mode by default (offline: hash embeddings + fake model).
# Point it at the deployed API with -e ASSISTANT_API_URL=... -e ASSISTANT_API_KEY=...
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY docs/ docs/
COPY scripts/ scripts/
COPY ui/ ui/
RUN python scripts/build_index.py --embedder hash

RUN useradd --create-home appuser
USER appuser

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "ui/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
