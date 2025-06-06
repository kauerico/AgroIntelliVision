# ==============================================================================
#   app.py
#
#   Backend do projeto AgroIntelliVision, implementado com Flask.
#   Responsável por receber imagens de folhas de soja, processá-las
#   e utilizar um modelo de machine learning para prever doenças.
#
#   Autor: Baseado no repositório de Katcilane Silva de Souza
#   Ajustes de integração: Gemini (Google)
# ==============================================================================

from flask import Flask, request, jsonify
from flask_cors import CORS  # Importa o CORS para permitir a comunicação entre domínios
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import os

# ==============================================================================
#   CORREÇÃO: Atualizado o caminho para corresponder à localização real do modelo.
# ==============================================================================
MODEL_PATH = os.path.join('models', 'saved_models', 'modelo_soja.h5')

# --- Carregamento do Modelo de Machine Learning ---
try:
    # Verifica se o ficheiro do modelo existe antes de tentar carregá-lo
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"O ficheiro do modelo '{MODEL_PATH}' não foi encontrado. Certifique-se de que está na pasta correta.")
    
    model = load_model(MODEL_PATH)
    print(">>> Modelo carregado com sucesso!")
    model.summary()  # Exibe um resumo da arquitetura do modelo no terminal
except Exception as e:
    print(f"!!! Erro fatal ao carregar o modelo: {e}")
    model = None # Garante que o modelo é None se o carregamento falhar

# --- Nomes das Classes ---
# IMPORTANTE: A ordem dos nomes deve ser EXATAMENTE a mesma usada durante o treino do modelo.
# A lista foi atualizada com nomes reais de doenças. Verifique se a ordem está correta!
class_names = [
    'Ferrugem Asiática',          # Classe 0
    'Mancha Alvo',                # Classe 1
    'Oídio',                      # Classe 2
    'Mancha Bacteriana',          # Classe 3
    'Míldio',                     # Classe 4
    'Crestamento Foliar de Cercospora', # Classe 5
    'Mancha Olho-de-Rã',          # Classe 6
    'Septoriose',                 # Classe 7
    'Antracnose',                 # Classe 8
    'Cancro da Haste',            # Classe 9
    'Podridão Parda da Haste',    # Classe 10
    'Mosaico Comum',              # Classe 11
    'Mancha Parda',               # Classe 12
    'Podridão de Sclerotinia',    # Classe 13
    'Deficiência de Potássio',    # Classe 14
    'Folha Saudável'              # Classe 15
]

def preprocess_image(image_bytes, target_size=(64, 64)):
    """
    Pré-processa a imagem recebida para o formato que o modelo espera.
    1. Abre a imagem a partir de bytes.
    2. Converte para o formato RGB.
    3. Redimensiona para o tamanho alvo.
    4. Converte para um array numpy e normaliza os pixels para o intervalo [0, 1].
    5. Adiciona uma dimensão extra para representar o "batch" (lote).
    """
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    image_array = np.asarray(image)
    image_array = image_array / 255.0  # Normalizar pixels
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# --- Configuração da Aplicação Flask ---
app = Flask(__name__)
# Habilita o CORS para permitir que o seu frontend (index.html) se comunique com este backend
CORS(app)

# --- Rota da API para Predição ---
@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint da API que recebe um ficheiro de imagem e retorna um diagnóstico.
    """
    # Verifica se o modelo foi carregado corretamente no início
    if model is None:
        return jsonify({'error': 'O modelo de machine learning não foi carregado corretamente no servidor.'}), 500

    # Verifica se um ficheiro foi enviado na requisição
    if 'file' not in request.files:
        return jsonify({'error': 'Nenhum ficheiro foi enviado na requisição.'}), 400

    file = request.files['file']

    # Verifica se o ficheiro tem um nome (evita envios vazios)
    if file.filename == '':
        return jsonify({'error': 'Ficheiro inválido ou sem nome.'}), 400

    try:
        # Lê os bytes da imagem e pré-processa
        image_bytes = file.read()
        processed_image = preprocess_image(image_bytes)
        
        # Realiza a predição com o modelo
        prediction = model.predict(processed_image)
        
        # Obtém o índice da classe com maior probabilidade e a confiança
        predicted_class_index = np.argmax(prediction[0])
        confidence = float(np.max(prediction[0]))
        
        # Mapeia o índice para o nome da classe
        if predicted_class_index < len(class_names):
            predicted_class_name = class_names[predicted_class_index]
        else:
            predicted_class_name = 'Classe Desconhecida'

        # Retorna o resultado em formato JSON
        return jsonify({
            'prediction': predicted_class_name,
            'confidence': confidence
        })

    except Exception as e:
        # Retorna um erro genérico se algo falhar durante o processo
        print(f"!!! Erro durante a predição: {e}")
        return jsonify({'error': f'Ocorreu um erro interno no servidor: {e}'}), 500

# ==============================================================================
#   Ponto de Entrada da Aplicação
#   Este bloco de código inicia o servidor web do Flask.
#   É a adição crucial que estava a faltar para que o seu backend ficasse online.
# ==============================================================================
if __name__ == '__main__':
    # app.run() inicia o servidor.
    # host='0.0.0.0' torna o servidor acessível na sua rede local.
    # port=5000 define a porta.
    # debug=True fornece mais detalhes sobre erros no terminal durante o desenvolvimento.
    app.run(host='0.0.0.0', port=5000, debug=True)
