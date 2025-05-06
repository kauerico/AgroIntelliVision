import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight
from config import settings
from models.build_model import build_model
from models.train import get_optimizer, compile_model
from utils.callbacks import get_callbacks

# Configurações para reduzir consumo de memória
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduz logs do TensorFlow

def verify_dataset(train_path, val_path):
    """Verifica a estrutura e balanceamento do dataset"""
    print("\n🔍 Verificando dataset...")
    
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise Exception("Estrutura inválida. Deve conter subpastas train/ e val/")

    class_counts = {}
    for split in ['train', 'val']:
        split_path = train_path if split == 'train' else val_path
        print(f"\n{split.upper()}:")
        for class_name in os.listdir(split_path):
            class_path = os.path.join(split_path, class_name)
            if os.path.isdir(class_path):
                num_images = len([f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
                class_counts[f"{split}_{class_name}"] = num_images
                print(f"{class_name}: {num_images} imagens")
                
                if split == 'train' and num_images < 100:
                    print(f"⚠️ Aviso: Classe {class_name} tem apenas {num_images} imagens (recomendado >=100)")
    return class_counts

def main():
    # 1. Configuração inicial
    print("⚙️ Configurando ambiente para CPU...")
    tf.config.set_visible_devices([], 'GPU')  # Força uso da CPU

    # 2. Caminhos do dataset
    train_path = os.path.join(settings.DATASET_PATH, 'train')
    val_path = os.path.join(settings.DATASET_PATH, 'val')
    
    # 3. Verificação do dataset
    class_counts = verify_dataset(train_path, val_path)

    # 4. Pré-processamento (lightweight para CPU)
    print("\n🔄 Preparando dados...")
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,       # Reduzido para CPU
        width_shift_range=0.1,   # Reduzido
        height_shift_range=0.1,  # Reduzido
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'      # Mais eficiente que 'reflect'
    )
    
    val_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    # 5. Carregamento dos dados
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

    # 6. Otimização de performance (ADICIONE AQUI)
    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    
    # Opcional: Se ainda houver OOM, adicione:
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    train_ds = train_ds.with_options(options)
    val_ds = val_ds.with_options(options)

    # 7. Class weights para dados desbalanceados
    print("\n⚖️ Calculando class weights...")
    class_weights = compute_class_weight('balanced',
                                       classes=np.unique(train_ds.classes),
                                       y=train_ds.classes)
    class_weights = dict(enumerate(class_weights))
    print("Class weights:", class_weights)

    # 8. Construção do modelo (leve para CPU)
    print("\n🛠️ Construindo modelo MobileNetV3 (otimizado para CPU)...")
    model = build_model()
    optimizer = get_optimizer(len(train_ds))
    model = compile_model(model, optimizer)
    model.summary()

    # 9. Callbacks com TensorBoard
    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint("best_model.keras", save_best_only=True),
        keras.callbacks.TensorBoard(log_dir='logs'),  # Para monitoramento
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )
    ]

    # 10. Treinamento (épocas reduzidas inicialmente)
    print(f"\n🎯 Iniciando treinamento ({settings.EPOCHS} épocas)...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=settings.EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    # 11. Salvamento do modelo
    model.save('modelo_soja_final_cpu.keras')
    print("\n✅ Modelo salvo como 'modelo_soja_final_cpu.keras'")

    # 12. Visualização de resultados
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Treino')
    plt.plot(history.history['val_accuracy'], label='Validação')
    plt.title('Acurácia')
    plt.xlabel('Época')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Treino')
    plt.plot(history.history['val_loss'], label='Validação')
    plt.title('Loss')
    plt.xlabel('Época')
    plt.legend()

    plt.tight_layout()
    plt.savefig('desempenho_treinamento_cpu.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    main()