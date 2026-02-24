# ---------- Base Image ----------
FROM python:3.10-slim

# Prevent python buffering
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# ---------- Workdir ----------
WORKDIR /app

# ---------- Install dependencies FIRST (important for cache) ----------
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# ---------- Copy project ----------
COPY . .

# ---------- Expose ----------
EXPOSE 10000

# ---------- Start server ----------
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]