from tensorflow import keras
from datetime import datetime
import os

def get_callbacks():
    # Cria diretório para logs
    log_dir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    
    return [
        keras.callbacks.EarlyStopping(
            patience=10,
            monitor='val_loss',
            restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            filepath='melhor_modelo.keras',
            save_best_only=True,
            monitor='val_loss'
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7
        ),
        # Adicione estes novos callbacks:
        keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            profile_batch='500,520'
        ),
        keras.callbacks.CSVLogger(
            filename='training_log.csv',
            separator=',',
            append=False
        )
    ]