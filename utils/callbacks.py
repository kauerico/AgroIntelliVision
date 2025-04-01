from tensorflow import keras

def get_callbacks():
    return [
        keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=12,
            mode='max',
            restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            'best_model.keras',
            monitor='val_auc',
            save_best_only=True,
            mode='max'
        ),
        keras.callbacks.TensorBoard(
            log_dir='./logs',
            histogram_freq=1
        )
    ]