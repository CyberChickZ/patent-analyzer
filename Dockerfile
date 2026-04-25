FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libffi-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps directly (faster than pip install . with build isolation)
RUN pip install --no-cache-dir \
    google-genai google-cloud-storage "fastapi>=0.115" "uvicorn[standard]>=0.30" \
    python-multipart pymupdf httpx sentence-transformers certifi numpy "a2a-sdk>=0.2" \
    "tenacity>=8.0"

# Pre-download embedding model so it's baked into the image
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY . .

EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
