"""## Criando um dataset

## Importando as bibliotecas
"""

# Bibliotecas para visualização de dados
import matplotlib.pyplot as plt  # Gráficos
import seaborn as sns  # Visualização estatística

# Bibliotecas para manipulação de arquivos e diretórios
import os
import shutil
import tarfile

# Bibliotecas para processamento de imagens
import cv2  # Visão computacional

# Bibliotecas para computação numérica e manipulação de dados
import numpy as np
import pandas as pd
from PIL import Image

# Bibliotecas para aprendizado de máquina e redes neurais
import tensorflow as tf
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Bibliotecas para avaliação de modelos
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split  # Divisão de datasets

# Verifica a versão do TensorFlow
print("TensorFlow version:", tf.__version__)

"""## Separando dataSet em treinamento, teste, validação"""

# Caminho para o dataset
dataset_path = 'C:/Users/ribei/Documents/ProjetoSoja/codigo/dados/DataSet'

# Lista de pastas de doenças
doencas = [
    'bean_rust', 'ferrugem_do_feijao', 'mancha_parda', 'podridao_radicular',
    'crestamento_bacteriano', 'mancha_alvo', 'mildio', 'saudavel',
    'deficiencia_de_potassio', 'mancha_angular', 'mosaico', 'sindrome_morte_subita',
    'ferrugem_asiatica', 'mancha_olho_de_ra', 'oidio', 'virus_mosaico'
]

# Listas para armazenar caminhos das imagens e rótulos
imagens = []
rotulos = []

# Coleta caminhos das imagens e rótulos
for doenca in doencas:
    pasta_doenca = os.path.join(dataset_path, doenca)
    if os.path.isdir(pasta_doenca):
        for imagem_nome in os.listdir(pasta_doenca):
            caminho_imagem = os.path.join(pasta_doenca, imagem_nome)
            if os.path.isfile(caminho_imagem):  # Apenas arquivos, não pastas
                imagens.append(caminho_imagem)
                rotulos.append(doenca)

# Divide os dados em treino, validação e teste
X_train_val, X_test, y_train_val, y_test = train_test_split(imagens, rotulos, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

print(f'Total de imagens válidas: {len(imagens)}')

"""# Configurando o gerador de imagens de treinamento"""

# Configurações do ImageDataGenerator para aumento de dados
gerador_treinamento = ImageDataGenerator(
    rescale=1./255,  # Normaliza os pixels para o intervalo [0, 1]
    rotation_range=7,  # Rotação aleatória de -7 a +7 graus
    width_shift_range=0.1,  # Deslocamento horizontal de até 10% da largura
    height_shift_range=0.1,  # Deslocamento vertical de até 10% da altura
    horizontal_flip=True,  # Inversão horizontal aleatória
    zoom_range=0.2  # Zoom aleatório de até 20%
)

"""### Carregando o conjunto de dados de treinamento usando o gerador de dados de imagem configurado"""

# Cria DataFrame com caminhos das imagens e rótulos
df_treinamento = pd.DataFrame({'Caminho': X_train, 'Rotulo': y_train})

# Configura o gerador de dados para o conjunto de treinamento
dataset_treinamento = gerador_treinamento.flow_from_dataframe(
    dataframe=df_treinamento,
    x_col='Caminho',          # Coluna com caminhos das imagens
    y_col='Rotulo',           # Coluna com rótulos
    target_size=(64, 64),     # Redimensiona imagens para 64x64
    batch_size=8,             # Tamanho do lote
    class_mode='categorical', # Modo de classificação multiclasse
    shuffle=True              # Embaralha as imagens
)

# Exibe o número de classes
num_classes = len(dataset_treinamento.class_indices)
print("Número de classes:", num_classes)

# Obtém o mapeamento de classes para índices
mapeamento_classes = dataset_treinamento.class_indices

# Exibe o mapeamento de classes
print("Mapeamento de classes:")
for classe, indice in mapeamento_classes.items():
    print(f"{classe}: {indice}")

# Configura o ImageDataGenerator para o conjunto de teste (apenas normalização)
gerador_teste = ImageDataGenerator(rescale=1./255)

# Cria DataFrame com caminhos das imagens e rótulos
df_teste = pd.DataFrame({'Caminho': X_test, 'Rotulo': y_test})

# Configura o gerador de dados para o conjunto de teste
dataset_teste = gerador_teste.flow_from_dataframe(
    dataframe=df_teste,
    x_col='Caminho',          # Coluna com caminhos das imagens
    y_col='Rotulo',           # Coluna com rótulos
    target_size=(64, 64),     # Redimensiona imagens para 64x64
    batch_size=1,             # Lote de tamanho 1 para previsões individuais
    class_mode='categorical', # Modo de classificação multiclasse
    shuffle=False             # Não embaralha para manter a ordem
)

# Exibe o número de classes
num_classes = len(dataset_teste.class_indices)
print("Número de classes:", num_classes)

# Cria uma rede neural sequencial
network = Sequential()

# Camadas convolucionais e de pooling
network.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(64,64,3)))
network.add(MaxPool2D(pool_size=(2,2)))
network.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
network.add(MaxPool2D(pool_size=(2,2)))
network.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
network.add(MaxPool2D(pool_size=(2,2)))

# Camadas densas (totalmente conectadas)
network.add(Flatten())
network.add(Dense(units=581, activation='relu'))
network.add(Dense(units=581, activation='relu'))

# Camada de saída
network.add(Dense(units=num_classes, activation='softmax'))

# Configura o modelo para treinamento
network.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Configura para carregar imagens truncadas
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Treina o modelo por 20 épocas
historico = network.fit(dataset_treinamento, epochs=10)

# Gera previsões para o conjunto de teste
previsoes = network.predict(dataset_teste)

# Converte previsões em classes
previsoes = np.argmax(previsoes, axis=1)

# Obtém os rótulos reais do conjunto de teste
rotulos_teste = dataset_teste.classes

# Avaliação do modelo
print("\nRelatório de Classificação:")
print(classification_report(rotulos_teste, previsoes))

print("\nAcurácia:", accuracy_score(rotulos_teste, previsoes))

# Matriz de confusão
cm = confusion_matrix(rotulos_teste, previsoes)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Matriz de Confusão')
plt.show()

# Salva o modelo
network.save('modelo_soja.keras')  # Formato moderno
network.save('modelo_soja.h5')     # Formato HDF5 (legado)

# Salva a arquitetura como JSON (opcional)
arquitetura_json = network.to_json()
with open('arquitetura_modelo.json', 'w') as json_file:
    json_file.write(arquitetura_json)