# -------------------------------
# Base Python image
# -------------------------------
FROM python:3.10-slim

# Prevent python from buffering logs
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set working directory
WORKDIR /app

# -------------------------------
# Install system deps (needed for numpy/faiss/etc)
# -------------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------
# Copy only requirements first (cache optimization)
# -------------------------------
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt

# -------------------------------
# Copy project files
# -------------------------------
COPY . /app

# -------------------------------
# Render provides PORT env variable
# -------------------------------
ENV PORT=10000

# -------------------------------
# Start FastAPI
# IMPORTANT: must bind 0.0.0.0 and use $PORT
# -------------------------------
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port $PORT"]