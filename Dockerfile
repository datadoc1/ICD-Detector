# Use slim Python base for smaller image
FROM python:3.10-slim

# Set working dir
WORKDIR /app

# Install system dependencies required for some Python packages (opencv, pillow, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    ca-certificates \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
 && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for better layer caching
COPY requirements.txt .

# Upgrade pip and install Python deps
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy repository
COPY . .

# Expose the port Gradio will listen on (internal container port)
EXPOSE 8080

# Ensure Gradio binds to 0.0.0.0 and uses the PORT provided by the platform
# Fly sets the $PORT env var automatically; we default to 8080 if not provided.
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=8080

# Start the app; ensure GRADIO_SERVER_PORT is set from $PORT at runtime if present
CMD ["sh", "-c", "export GRADIO_SERVER_PORT=${PORT:-8080} && python app.py"]