import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import base64
from io import BytesIO
import json
import argparse

# Configurações
plt.style.use('ggplot')

def parse_args():
    parser = argparse.ArgumentParser(description='Avaliador de Modelos de Classificação')
    parser.add_argument('--model-path', type=str, required=True, help='Caminho para o modelo .keras')
    parser.add_argument('--dataset-dir', type=str, required=True, help='Diretório com as imagens de validação')
    parser.add_argument('--output-dir', type=str, default='/reports', help='Diretório de saída para os relatórios')
    return parser.parse_args()

def load_model(model_path):
    """Carrega o modelo TensorFlow com verificação de compatibilidade"""
    try:
        model = tf.keras.models.load_model(model_path)
        # Verificação de saúde do modelo
        if not hasattr(model, 'predict'):
            raise ValueError("Modelo não possui método predict()")
        return model
    except Exception as e:
        raise RuntimeError(f"Falha ao carregar modelo: {str(e)}")

def load_and_preprocess_image(image_path, target_size):
    """Carrega e pré-processa a imagem"""
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Adiciona a dimensão do batch
    img_array /= 255.0  # Normaliza os valores dos pixels
    return img_array

def generate_html_report(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_filename = f"relatorio_classificacao_{timestamp}.html"
    report_path = os.path.join(output_dir, report_filename)

    cm = confusion_matrix(results['true_labels'], results['predicted_labels'])
    class_names = list(results['classification_report'].keys())[:-3]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="YlGnBu")
    plt.xlabel("Predição feita pelo modelo")
    plt.ylabel("Categoria real")
    plt.title("Matriz de Confusão")

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    plt.close()
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()

    df_report = pd.DataFrame(results['classification_report']).transpose().round(2)
    df_report = df_report.rename(columns={
        "precision": "Precisão (quantas predições estavam certas)",
        "recall": "Abrangência (quanto foi detectado corretamente)",
        "f1-score": "Equilíbrio entre precisão e abrangência",
        "support": "Número de exemplos"
    })

    acertos_totais = np.sum(np.array(results['true_labels']) == np.array(results['predicted_labels']))
    total = len(results['true_labels'])
    precisao_geral = round(acertos_totais / total * 100, 2)

    html = f"""
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Relatório de Classificação</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background-color: #f9f9f9;
                margin: 20px;
                color: #333;
            }}
            h1, h2 {{
                color: #2c3e50;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin-top: 20px;
            }}
            th, td {{
                border: 1px solid #ccc;
                padding: 10px;
                text-align: center;
            }}
            th {{
                background-color: #e8f4f8;
            }}
            img {{
                max-width: 100%;
                height: auto;
                margin-top: 20px;
                border: 1px solid #ccc;
            }}
            .resumo {{
                background-color: #ecf9ec;
                border: 1px solid #c3e6c3;
                padding: 15px;
                margin-top: 20px;
                border-radius: 5px;
            }}
        </style>
    </head>
    <body>
        <h1>Relatório de Avaliação do Modelo</h1>

        <div class="resumo">
            <p><strong>Total de imagens avaliadas:</strong> {total}</p>
            <p><strong>Acertos:</strong> {acertos_totais}</p>
            <p><strong>Precisão geral:</strong> {precisao_geral}%</p>
            <p>Este relatório mostra como o modelo se saiu ao classificar imagens de diferentes doenças em plantas.</p>
        </div>

        <h2>Matriz de Confusão</h2>
        <p>Ela mostra como o modelo confundiu uma categoria com outra. O ideal é que os números fiquem na diagonal.</p>
        <img src="data:image/png;base64,{image_base64}" alt="Matriz de Confusão">

        <h2>Métricas por Categoria</h2>
        <p>Abaixo, cada linha representa uma categoria de doença. Veja como o modelo se saiu em cada uma.</p>
        {df_report.to_html(classes='table table-striped', border=0)}
    </body>
    </html>
    """

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html)

    return report_path

def evaluate_model(model, dataset_dir, class_names):
    """Avaliação robusta com tratamento de erros"""
    try:
        target_size = model.input_shape[1:3]
        true_labels, predicted_labels = [], []
        
        # Coleta de dados com progresso
        for class_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(dataset_dir, class_name)
            if not os.path.exists(class_dir):
                continue

            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Ajuste os tipos de arquivo conforme necessário
                    img_path = os.path.join(class_dir, img_file)
                    try:
                        img = load_and_preprocess_image(img_path, target_size)
                        prediction = model.predict(img, verbose=0)
                        predicted_idx = np.argmax(prediction, axis=1)[0]

                        true_labels.append(class_idx)
                        predicted_labels.append(predicted_idx)
                    except Exception as e:
                        print(f"Erro ao processar {img_file}: {e}")

        # Gerar o relatório de classificação
        report = classification_report(true_labels, predicted_labels, target_names=class_names, output_dict=True)
        return {
            "true_labels": true_labels,
            "predicted_labels": predicted_labels,
            "classification_report": report
        }

    except Exception as e:
        raise RuntimeError(f"Erro na avaliação do modelo: {str(e)}")

def main():
    args = parse_args()
    try:
        model = load_model(args.model_path)
        class_names = os.listdir(args.dataset_dir)  # Lista as classes no diretório

        results = evaluate_model(model, args.dataset_dir, class_names)
        report_path = generate_html_report(results, args.output_dir)
        print(f"Relatório gerado com sucesso em: {report_path}")

    except Exception as e:
        print(f"Erro durante a execução: {e}")

if __name__ == '__main__':
    main()
