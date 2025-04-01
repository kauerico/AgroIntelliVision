import tensorflow as tf
from tensorflow import keras
from config import settings

def create_data_flow(subset):
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=0.15,
        rotation_range=45,
        width_shift_range=0.3,
        height_shift_range=0.3,
        brightness_range=[0.7,1.3],
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        channel_shift_range=0.2,
        fill_mode='reflect'
    )
    
    return train_datagen.flow_from_directory(
        settings.DATASET_PATH,
        target_size=settings.IMG_SIZE,
        batch_size=settings.BATCH_SIZE,
        subset=subset,
        class_mode='sparse',
        shuffle=True,
        seed=42
    )