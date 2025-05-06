import os
import tensorflow as tf
from tensorflow import keras
from config import settings

def create_data_flow(subset):
    if subset == 'training':
        datagen = keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        directory = os.path.join(settings.DATASET_PATH, 'train')
        shuffle = True
    else:
        datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        directory = os.path.join(settings.DATASET_PATH, 'val')
        shuffle = False

    return datagen.flow_from_directory(
        directory,
        target_size=settings.IMG_SIZE,
        batch_size=settings.BATCH_SIZE,
        class_mode='categorical',
        shuffle=shuffle,
        seed=42
    )
