# syntax=docker/dockerfile:1.4.3
FROM python:3.12-slim-bookworm

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the sentence-transformer weights at build time: otherwise the
# first Semantic-mode run pays the download, and the container can't do
# semantic clustering at all without network access. config.py is copied
# first (it's just constants, no imports) so the model name stays
# single-sourced instead of being repeated here.
COPY config.py .
RUN python -c "from config import SEMANTIC_EMBEDDING_MODEL; \
from sentence_transformers import SentenceTransformer; \
SentenceTransformer(SEMANTIC_EMBEDDING_MODEL)"

COPY . .

EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "app.py"]
