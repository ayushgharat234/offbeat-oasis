FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app:/app/src

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libatlas-base-dev \
    liblapack-dev \
    gfortran \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy files
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app/ app/
COPY src/ src/
COPY data/ data/
COPY logs/ logs/

# Make logs writable
RUN mkdir -p logs && chmod -R 777 logs

EXPOSE 8501

CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]