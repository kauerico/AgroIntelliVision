FROM python:3.11-slim

WORKDIR /app

COPY requirements-github.txt .
RUN pip install --upgrade pip && pip install -r requirements-github.txt

COPY models/test_models/ models/test_models/

# Entrada que aceita argumentos (modelos e imagens)
CMD ["python", "models/test_models/main.py", "/models", "/imagens"]
