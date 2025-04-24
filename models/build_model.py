from tensorflow import keras
from tensorflow.keras import layers
from config import settings

def build_model():
    # Usando transfer learning com EfficientNet
    base_model = keras.applications.EfficientNetV2B0(  # B0 é mais leve que B2
        input_shape=(*settings.IMG_SIZE, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    base_model.trainable = False

    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(*settings.IMG_SIZE, 3)),
        keras.layers.Rescaling(1./255),  # Normalização incluída no modelo
        base_model,
        layers.Dropout(0.5),  # Reduzi dropout para evitar underfitting
        layers.Dense(256, activation='swish', kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dense(128, activation='swish', kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.Dropout(0.3),
        layers.Dense(settings.NUM_CLASSES, activation='softmax')
    ])
    
    return model