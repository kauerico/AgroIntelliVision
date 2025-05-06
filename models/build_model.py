from tensorflow import keras
from tensorflow.keras import layers
from config import settings

def build_model():
    base_model = keras.applications.EfficientNetB0(
        input_shape=(*settings.IMG_SIZE, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    
    # Congela as camadas iniciais
    for layer in base_model.layers[:100]:
        layer.trainable = False
        
    inputs = keras.Input(shape=(*settings.IMG_SIZE, 3))
    x = keras.layers.Rescaling(1./255)(inputs)
    x = base_model(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    outputs = keras.layers.Dense(settings.NUM_CLASSES, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    return model
