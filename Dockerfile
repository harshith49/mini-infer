FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 HF_HOME=/app/model_cache
COPY requirements.txt .
RUN python -m pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "-m", "server.app", "--host", "0.0.0.0", "--port", "8000", "--device", "cpu", "--threads", "1"]
