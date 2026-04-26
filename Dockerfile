# ══════════════════════════════════════════════════════════════
# LifeOS Agent — Production Dockerfile
# ══════════════════════════════════════════════════════════════
# Default: runs the OpenEnv FastAPI server on port 8000
# Alt:     runs the Gradio UI on port 7860 (see bottom)
# ══════════════════════════════════════════════════════════════

FROM python:3.10-slim

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY models.py .
COPY client.py .
COPY app_ui.py .
COPY openenv.yaml .
COPY __init__.py .
COPY server/ ./server/

# Expose ports (8000 = FastAPI, 7860 = Gradio)
EXPOSE 8000
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Default: run the OpenEnv FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]

# ──────────────────────────────────────────────────────────────
# To run the Gradio UI instead, override the CMD:
#
#   docker build -t lifeos-agent .
#   docker run -p 7860:7860 lifeos-agent python app_ui.py
#
# Or for the FastAPI server (default):
#
#   docker build -t lifeos-agent .
#   docker run -p 8000:8000 lifeos-agent
#
# Both at once (two containers):
#
#   docker run -d -p 8000:8000 --name lifeos-server lifeos-agent
#   docker run -d -p 7860:7860 --name lifeos-ui lifeos-agent python app_ui.py
# ──────────────────────────────────────────────────────────────
