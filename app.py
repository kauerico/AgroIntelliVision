from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Habilita CORS para todas as rotas

# Carrega o modelo
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'modelo_soja.keras')

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Modelo carregado com sucesso!")
    print("\nResumo do modelo:")
    model.summary()
except Exception as e:
    print(f"Erro ao carregar modelo: {e}")
    raise

# Mapeamento de classes (ajuste conforme seu dataset)
class_names = [
    "ferrugem_asiatica",
    "mancha_alvo",
    "mancha_angular",
    "mancha_parda",
    "mildio",
    "podridao_radicular",
    "sindrome_morte_subita",
    "saudavel"
]



@app.route('/')
def home():
    return "API de Predição de Doenças na Soja - AgroIntelliVision"

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'Nenhuma imagem enviada'}), 400
    
    try:
        # Carrega e pré-processa a imagem
        img = Image.open(io.BytesIO(request.files['image'].read()))
        img = img.convert('RGB')
        
        # Redimensiona conforme o modelo espera (256x256)
        img = img.resize(settings.IMG_SIZE)
        
        # Converte para array e normaliza
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Faz a predição
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions[0])
        
        # Prepara a resposta
        response = {
            'predicted_class': class_names[predicted_index],
            'confidence': float(predictions[0][predicted_index]),
            'all_predictions': [
                {'class': name, 'confidence': float(conf)}
                for name, conf in zip(class_names, predictions[0])
            ],
            'top3_predictions': sorted(
                [{'class': name, 'confidence': float(conf)} 
                 for name, conf in zip(class_names, predictions[0])],
                key=lambda x: x['confidence'], reverse=True
            )[:3]
        }
        
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500