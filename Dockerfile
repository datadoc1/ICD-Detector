# Use slim Python base for smaller image
FROM python:3.10-slim

# Set working dir
WORKDIR /app

# Install minimal system dependencies required for runtime (opencv, pillow)
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    ca-certificates \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
 && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for better layer caching
COPY requirements.txt .

# Upgrade pip and install Python dependencies (use CPU-only PyTorch for smaller image)
RUN pip install --upgrade pip setuptools wheel && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy model artifact explicitly so it's present in the image even if a .dockerignore exists
# (placed before copying the rest of the repo to ensure it gets included)
COPY model ./model

# Copy repository
COPY . .

# Expose the port Gradio will listen on (internal container port)
EXPOSE 8080

# Ensure Gradio binds to 0.0.0.0 and use platform PORT at runtime
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=8080

# Start the app; ensure GRADIO_SERVER_PORT is set from $PORT at runtime if present
CMD ["sh", "-c", "export GRADIO_SERVER_PORT=${PORT:-8080} && python app.py"]