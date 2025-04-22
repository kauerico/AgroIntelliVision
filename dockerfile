# Usando a versão slim do TensorFlow
FROM tensorflow/tensorflow:latest AS build

# Diretório de trabalho dentro do container
WORKDIR /app

# Copiar o requirements
COPY requirements-github.txt .

# Instalar as dependências
RUN pip install --upgrade pip && \
    pip install -r requirements-github.txt

# Copiar o código
COPY models/test_models/ models/test_models/

# Começar o estágio final
FROM tensorflow/tensorflow:latest

# Copiar as dependências do build (só as necessárias)
COPY --from=build /app /app

WORKDIR /app

# Comando de entrada
CMD ["python", "models/test_models/main.py", "/models", "/imagens"]
