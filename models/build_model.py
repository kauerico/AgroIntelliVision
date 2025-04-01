from tensorflow import keras
from tensorflow.keras import layers
from config import settings

def build_model():
    base_model = keras.applications.EfficientNetV2B2(
        input_shape=(*settings.IMG_SIZE, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    base_model.trainable = False

    return keras.Sequential([
        base_model,
        layers.Dropout(0.6),
        layers.Dense(512, activation='swish', kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dense(256, activation='swish', kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.Dropout(0.4),
        layers.Dense(settings.NUM_CLASSES, activation='softmax', dtype=tf.float32)
    ])