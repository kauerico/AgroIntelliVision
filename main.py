import tensorflow as tf
from tensorflow import keras
import numpy as np
from config import settings
from data.visualization import plot_class_distribution
from models.build_model import build_model
from models.train import get_optimizer, compile_model
from utils.callbacks import get_callbacks

# Verifica se o diretório de treino existe
# Se não existir, levanta uma exceção
# para evitar erros mais tarde
# e facilitar o debugging
import os
train_path = os.path.join(settings.DATASET_PATH, 'train')
if not os.path.exists(train_path):
    raise Exception(f"Pasta de treino não encontrada em: {train_path}")

def main():
    # 1. Configuração inicial
    if settings.MIXED_PRECISION and tf.config.list_physical_devices('GPU'):
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print('Mixed precision enabled')

    # 2. Pré-processamento com aumento de dados
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=25,  # Aumentado
        width_shift_range=0.15,  # Reduzido para evitar distorções muito grandes
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,  # Adicionado
        fill_mode='reflect'  # Melhor para imagens naturais
    )
    
    val_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    # 3. Carregando os dados
    train_ds = train_datagen.flow_from_directory(
        f'{settings.DATASET_PATH}/train',
        target_size=settings.IMG_SIZE,
        batch_size=settings.BATCH_SIZE,
        class_mode='sparse',
        shuffle=True,
        seed=42  # Para reprodutibilidade
    )
    import matplotlib.pyplot as plt
    import numpy as np

    print("\nDistribuição das classes de treino:")
    plt.figure(figsize=(10, 5))
    plt.bar(list(train_ds.class_indices.keys()), np.bincount(train_ds.classes))
    plt.title('Distribuição das Classes (Treino)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    val_ds = val_datagen.flow_from_directory(
        f'{settings.DATASET_PATH}/val',
        target_size=settings.IMG_SIZE,
        batch_size=settings.BATCH_SIZE,
        class_mode='sparse',
        shuffle=False  # Importante para validação
    )

    # 4. Visualização da distribuição das classes
    plot_class_distribution(train_ds.class_indices, np.bincount(train_ds.classes))

    # 5. Construção do modelo
    model = build_model()
    optimizer = get_optimizer(len(train_ds))
    model = compile_model(model, optimizer)
    model.summary()

    # 6. Treinamento inicial
    print("\nIniciando treinamento base...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=settings.EPOCHS,
        callbacks=get_callbacks(),
        verbose=1
    )

    # 7. Fine-tuning
    print("\nIniciando fine-tuning...")
    model.get_layer('efficientnetv2-b0').trainable = True
    
    # Otimizador com learning rate menor para fine-tuning
    fine_tune_optimizer = keras.optimizers.Adam(
        learning_rate=1e-5,  # LR menor que o inicial
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07
    )
    
    model.compile(
        optimizer=fine_tune_optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=model.compiled_metrics.metrics
    )
    
    history_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=settings.EPOCHS + 15,  # Aumentei +15 epochs para fine-tuning
        initial_epoch=history.epoch[-1] + 1,
        callbacks=get_callbacks(),
        verbose=1
    )

    # 8. Salvamento do modelo
    print("\nSalvando modelo final...")
    model.save('modelo_soja_final.keras')
    
    # 9. Avaliação final
    print("\nAvaliando modelo final...")
    eval_results = model.evaluate(val_ds, verbose=1)
    print(f"Resultados finais - Loss: {eval_results[0]}, Accuracy: {eval_results[1]}")

if __name__ == "__main__":
    main()