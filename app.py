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
        
        # Redimensiona conforme o modelo espera (64x64 conforme o erro indicado)
        img = img.resize((64, 64))
        
        # Converte para array e normaliza
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Verificação de forma
        print(f"Forma da imagem processada: {img_array.shape}")
        print(f"Forma esperada pelo modelo: {model.input_shape}")
        
        if img_array.shape[1:] != model.input_shape[1:]:
            return jsonify({
                'error': f'Dimensões inválidas. Recebido: {img_array.shape[1:]}, Esperado: {model.input_shape[1:]}'
            }), 400
        
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
            ]
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)