FROM arm64v8/alpine:latest

RUN apk --no-cache add python3

# Install system dependencies
RUN apk add --no-cache \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create required directories
RUN mkdir -p ./record ./record_mosaic ./record_encrypt

# Download YOLO face detection model
RUN pip install --no-cache-dir ultralytics && \
    python -c "from ultralytics import YOLO; YOLO('yolov11n-face.pt')"

# Expose port
EXPOSE 52049

# Run the application
CMD ["python", "main.py"]
