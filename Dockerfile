FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    OMP_NUM_THREADS=4 \
    PORT=5000

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt

# This research service runs CPU inference in Compose. Avoid downloading CUDA
# libraries into the image; keep the torch/torchvision versions paired.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && pip install --no-cache-dir -r requirements.txt

COPY app.py ./app.py
COPY chatbot/ ./chatbot/
COPY templates/ ./templates/
COPY static/ ./static/
COPY models/ ./models/
COPY scaler.pkl ./scaler.pkl

RUN mkdir -p /app/instance /app/static/uploads /app/static/reports

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=5 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/api/health' % os.getenv('PORT', '5000'), timeout=5)" || exit 1

CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 1 --threads 4 --timeout 300 --preload --access-logfile - --error-logfile - app:app"]
