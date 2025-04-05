import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from config.settings import EPOCHS
from data.preprocessing import create_data_flow
from data.visualization import plot_class_distribution
from models.build_model import build_model
from models.train import get_optimizer, compile_model
from utils.callbacks import get_callbacks

def main():
    # Configuração inicial
    if tf.config.list_physical_devices('GPU'):
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    # Pré-processamento de dados
    train_ds = create_data_flow('training') 
    val_ds = create_data_flow('validation')
    
    # Visualização de dados
    plot_class_distribution(list(train_ds.class_indices.keys()), np.bincount(train_ds.classes))
    
    # Construção do modelo
    model = build_model()
    optimizer = get_optimizer(len(train_ds))
    model = compile_model(model, optimizer)
    model.summary()
    
    # Treinamento
    callbacks = get_callbacks()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    
    # Fine-tuning
    model.get_layer('efficientnetv2-b2').trainable = True
    model.compile(
        optimizer=keras.optimizers.Adam(1e-6),
        loss='sparse_categorical_crossentropy',
        metrics=model.compiled_metrics.metrics
    )
    
    history_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS + 10,
        initial_epoch=history.epoch[-1],
        callbacks=callbacks
    )

    # Diretório base onde os modelos serão salvos
    base_path = './models/saved_models'

    # Criar diretório base se não existir
    os.makedirs(base_path, exist_ok=True)

    # Pegar os números já usados nas pastas
    existing = [int(name) for name in os.listdir(base_path) if name.isdigit()]
    next_tag = max(existing) + 1 if existing else 1

    # Caminho final para salvar
    save_path = os.path.join(base_path, str(next_tag))
    model.save(save_path)
    print(f"Modelo salvo em: {save_path}")


if __name__ == "__main__":
    main()