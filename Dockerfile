FROM python:3.9-alpine

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY templates/ ./templates/


CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8086", "--reload"]