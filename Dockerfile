FROM runpod/base:0.6.2-cuda12.1.0

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir onnxruntime-gpu runpod && \
    pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Pre-download insightface model at build time
RUN python3 -c "import insightface; m = insightface.app.FaceAnalysis(providers=['CPUExecutionProvider']); m.prepare(ctx_id=-1)"

CMD ["python3", "handler.py"]
