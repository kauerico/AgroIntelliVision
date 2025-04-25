import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
from config import settings
from data.visualization import plot_class_distribution
from models.build_model import build_model
from models.train import get_optimizer, compile_model
from utils.callbacks import get_callbacks

def main():
    # 1. Configuração inicial
    if settings.MIXED_PRECISION and tf.config.list_physical_devices('GPU'):
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print('✅ Mixed precision habilitado')

    # 2. Verificação das pastas do dataset
    train_path = os.path.join(settings.DATASET_PATH, 'train')
    val_path = os.path.join(settings.DATASET_PATH, 'val')
    
    if not os.path.exists(train_path):
        raise Exception(f"❌ Pasta de treino não encontrada em: {train_path}\n"
                      f"Certifique-se que:\n"
                      f"- O caminho em settings.py está correto\n"
                      f"- Existe uma pasta 'train' com subpastas para cada classe")
    
    # 3. Pré-processamento com aumento de dados
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='reflect'
    )
    
    val_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    # 4. Carregando os dados
    print("\n📂 Carregando dataset...")
    train_ds = train_datagen.flow_from_directory(
    train_path,
    target_size=settings.IMG_SIZE,
    batch_size=settings.BATCH_SIZE,
    class_mode='categorical',  # Mude de 'sparse' para 'categorical'
    shuffle=True,
    seed=42
)

    val_ds = val_datagen.flow_from_directory(
    val_path,
    target_size=settings.IMG_SIZE,
    batch_size=settings.BATCH_SIZE,
    class_mode='categorical',  # Aqui também
    shuffle=False
)

    # 5. Visualização da distribuição das classes
    print("\n📊 Gerando gráfico de distribuição...")
    plt.figure(figsize=(12, 6))
    bars = plt.bar(list(train_ds.class_indices.keys()), 
                  np.bincount(train_ds.classes),
                  color='#4c72b0')
    
    # Adiciona valores nas barras
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    plt.title(f'Distribuição de Imagens por Classe\nTotal: {sum(np.bincount(train_ds.classes))} imagens', pad=20)
    plt.xlabel('Classes')
    plt.ylabel('Quantidade')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()

    # Salva o gráfico
    os.makedirs(os.path.join('images', 'graficos'), exist_ok=True)
    graph_path = os.path.join('images', 'graficos', 
                            f'distribuicao_classes_{datetime.now().strftime("%Y%m%d")}.png')
    plt.savefig(graph_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✔ Gráfico salvo em: {graph_path}")

    # 6. Construção do modelo
    print("\n🛠️ Construindo modelo...")
    model = build_model()
    optimizer = get_optimizer(len(train_ds))
    model = compile_model(model, optimizer)
    model.summary()

    # 7. Treinamento
    print("\n🎯 Iniciando treinamento...")
   # Substitua a linha do model.fit() por:
    history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=settings.EPOCHS,
    callbacks=get_callbacks(),
    verbose=1,
    workers=4,  # Adicione estas 3 linhas
    use_multiprocessing=True,
    max_queue_size=10
)

    # 8. Fine-tuning
    print("\n🔧 Ajuste fino (fine-tuning)...")
    model.get_layer('efficientnetv2-b0').trainable = True
    
    fine_tune_optimizer = keras.optimizers.Adam(
        learning_rate=1e-5,
        beta_1=0.9,
        beta_2=0.999
    )
    
    model.compile(
        optimizer=fine_tune_optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=model.compiled_metrics.metrics
    )
    
    history_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=settings.EPOCHS + 15,
        initial_epoch=history.epoch[-1] + 1,
        callbacks=get_callbacks(),
        verbose=1
    )

    # 9. Salvamento do modelo
    model.save('modelo_soja_final.keras')
    print("\n✅ Modelo salvo como 'modelo_soja_final.keras'")

if __name__ == "__main__":
    main()