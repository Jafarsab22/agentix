# Dockerfile
FROM python:3.11-slim

# System deps for Chromium
RUN apt-get update && apt-get install -y --no-install-recommends \
    chromium chromium-driver \
    fonts-liberation fonts-dejavu-core \
    libglib2.0-0 libnss3 libgdk-pixbuf2.0-0 libgtk-3-0 \
    libx11-6 libxcomposite1 libxdamage1 libxrandr2 libxshmfence1 \
    libasound2 ca-certificates && \
    rm -rf /var/lib/apt/lists/*

ENV CHROME_BIN=/usr/bin/chromium
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

# Railway sets PORT; Gradio binds to it in app.py
EXPOSE 8080
CMD ["python", "app.py"]
