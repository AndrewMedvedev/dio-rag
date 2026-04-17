FROM python:3.13-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends git build-essential && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

WORKDIR /app

# Копируем только requirements сначала для лучшего кэширования
COPY ./fastapi-rag/requirements.txt .

RUN pip install --upgrade pip

# Устанавливаем только необходимые пакеты
RUN pip install --no-cache-dir -r requirements.txt

# Удаляем ненужные build dependencies
RUN apt-get remove -y build-essential && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

COPY fastapi-rag/ .

CMD ["python", "main.py"]