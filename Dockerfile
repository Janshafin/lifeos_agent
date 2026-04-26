FROM python:3.10-slim

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

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

# HuggingFace Spaces requires port 7860
EXPOSE 7860

# Run the Gradio UI
CMD ["python", "app_ui.py"]
