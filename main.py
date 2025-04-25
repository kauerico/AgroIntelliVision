import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
from config import settings
from models.build_model import build_model
from models.train import get_optimizer, compile_model
from utils.callbacks import get_callbacks

# Desativa mensagens verbosas do oneDNN
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def main():
    # 1. Configuração inicial
    print("⚙️ Configurando ambiente...")
    if settings.MIXED_PRECISION and tf.config.list_physical_devices('GPU'):
        policy = keras.mixed_precision.Policy('mixed_float16')
        keras.mixed_precision.set_global_policy(policy)
        print("✅ Precisão mista ativada")

    # 2. Verificação da estrutura do dataset
    print("\n🔍 Verificando estrutura do dataset...")
    train_path = os.path.join(settings.DATASET_PATH, 'train')
    val_path = os.path.join(settings.DATASET_PATH, 'val')
    
    if not os.path.exists(train_path):
        raise Exception(f"❌ Pasta de treino não encontrada em: {train_path}\n"
                      f"Estrutura esperada:\n"
                      f"{settings.DATASET_PATH}/\n"
                      f"├── train/\n"
                      f"│   ├── classe1/\n"
                      f"│   └── classe2/\n"
                      f"└── val/\n"
                      f"    ├── classe1/\n"
                      f"    └── classe2/")

    # 3. Pré-processamento dos dados
    print("\n🔄 Preparando dados...")
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='reflect',
        brightness_range=[0.8, 1.2]  # Adicionado variação de brilho
    )
    
    # Gerador para dados de validação (sem aumento de dados)
    val_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    # 4. Carregamento dos dados
    print("\n📂 Carregando imagens...")
    train_ds = train_datagen.flow_from_directory(
        train_path,
        target_size=settings.IMG_SIZE,
        batch_size=settings.BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=42
    )
    
    val_ds = val_datagen.flow_from_directory(
        val_path,
        target_size=settings.IMG_SIZE,
        batch_size=settings.BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    # 5. Verificação do formato dos dados
    print("\n🔍 Verificando formato dos dados:")
    try:
        # Pega o primeiro batch de dados
        imagens, rotulos = next(train_ds)
        
        print(f"Formato das imagens: {imagens.shape}")
        print(f"Formato dos rótulos: {rotulos.shape}")
        print(f"Exemplo de rótulo: {rotulos[0]}")
        
        # Visualiza uma imagem de exemplo
        plt.figure(figsize=(6,6))
        plt.imshow(imagens[0])
        plt.title(f"Classe: {np.argmax(rotulos[0])}")
        plt.axis('off')
        
        # Salva a visualização
        os.makedirs(os.path.join('images', 'graficos'), exist_ok=True)
        plt.savefig(os.path.join('images', 'graficos', 'imagem_exemplo.png'), 
                   bbox_inches='tight', dpi=150)
        plt.close()
        print("✅ Verificação dos dados concluída com sucesso")
        
    except Exception as e:
        print(f"❌ Falha na verificação: {str(e)}")
        raise

    # 6. Visualização da distribuição das classes
    print("\n📊 Gerando gráfico de distribuição das classes...")
    # Método robusto para contar as classes
    contagem_classes = np.zeros(len(train_ds.class_indices))
    samples = 0
    
    for i, (_, labels) in enumerate(train_ds):
        contagem_classes += np.sum(labels, axis=0)
        samples += labels.shape[0]
        
        if i >= len(train_ds) - 1:  # Quando completa um epoch
            break
    
    total_imagens = int(np.sum(contagem_classes))
    class_names = list(train_ds.class_indices.keys())

    plt.figure(figsize=(12,6))
    barras = plt.bar(class_names, contagem_classes, color='#1f77b4')

    # Adiciona valores nas barras
    for barra in barras:
        altura = barra.get_height()
        plt.text(barra.get_x() + barra.get_width()/2., altura,
                f'{int(altura)}',
                ha='center', va='bottom')
    
    plt.title(f'Distribuição das Classes (Total: {total_imagens} imagens)', pad=20)
    plt.xlabel('Classes')
    plt.ylabel('Quantidade')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()

    # Salva o gráfico de distribuição
    os.makedirs(os.path.join('images', 'graficos'), exist_ok=True)
    caminho_grafico = os.path.join('images', 'graficos', 
                                 f'distribuicao_classes_{datetime.now().strftime("%Y%m%d_%H%M")}.png')
    plt.savefig(caminho_grafico, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Gráfico salvo em: {caminho_grafico}")

    # 7. Construção do modelo
    print("\n🛠️ Construindo o modelo...")
    modelo = build_model()
    otimizador = get_optimizer(len(train_ds))
    modelo = compile_model(modelo, otimizador)
    modelo.summary()

    # 8. Treinamento
    print(f"\n🎯 Iniciando treinamento ({settings.EPOCHS} épocas)...")
    historico = modelo.fit(
        train_ds,
        validation_data=val_ds,
        epochs=settings.EPOCHS,
        callbacks=get_callbacks(),
        verbose=1,
        
    )

    # 9. Fine-tuning (ajuste fino)
    print("\n🔧 Aplicando fine-tuning...")
    modelo.get_layer('efficientnetv2-b0').trainable = True
    modelo.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss='categorical_crossentropy',
        metrics=modelo.compiled_metrics.metrics
    )
    
    historico_fine = modelo.fit(
        train_ds,
        validation_data=val_ds,
        epochs=settings.EPOCHS + 10,
        initial_epoch=historico.epoch[-1] + 1,
        callbacks=get_callbacks(),
        verbose=1
    )

    # 10. Salvamento do modelo
    modelo.save('modelo_soja_final.keras')
    print("\n✅ Modelo treinado salvo como 'modelo_soja_final.keras'")

    # 11. Plots de desempenho
    print("\n📈 Gerando gráficos de desempenho...")
    plt.figure(figsize=(12, 6))
    
    # Gráfico de acurácia
    plt.subplot(1, 2, 1)
    plt.plot(historico.history['accuracy'], label='Treino')
    plt.plot(historico.history['val_accuracy'], label='Validação')
    plt.title('Acurácia durante o Treinamento')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.legend()
    
    # Gráfico de loss
    plt.subplot(1, 2, 2)
    plt.plot(historico.history['loss'], label='Treino')
    plt.plot(historico.history['val_loss'], label='Validação')
    plt.title('Loss durante o Treinamento')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    caminho_desempenho = os.path.join('images', 'graficos', 'desempenho_treinamento.png')
    plt.savefig(caminho_desempenho, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Gráficos de desempenho salvos em: {caminho_desempenho}")

if __name__ == "__main__":
    main()