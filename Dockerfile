# Use slim Python base for smaller image
FROM python:3.10-slim

# Set working dir
WORKDIR /app

# Install minimal system dependencies required for runtime (opencv, pillow)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .

# Upgrade pip and install all Python dependencies in one layer
# torch torchvision are CPU-only to keep image small
# --no-cache-dir avoids caching large wheels in image layers
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy repository (model/, app.py, white_paper.md, etc.)
COPY . .

# Expose the port Gradio will listen on (internal container port)
EXPOSE 8080

# Ensure Gradio binds to 0.0.0.0 and use platform PORT at runtime
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=8080

# Start the app; ensure GRADIO_SERVER_PORT is set from $PORT at runtime if present
CMD ["sh", "-c", "export GRADIO_SERVER_NAME=0.0.0.0 && export GRADIO_SERVER_PORT=${PORT:-8080} && python app.py"]
