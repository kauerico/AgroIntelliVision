import tensorflow as tf
import os

BATCH_SIZE = 16  # Reduza para evitar estouro de memória (8 se ainda der problemas)
IMG_SIZE = (224, 224)  # Tamanho menor = menos RAM
EPOCHS = 10  # Comece com menos épocas (aumente depois se necessário)
NUM_CLASSES = 8
AUTOTUNE = tf.data.AUTOTUNE
MIXED_PRECISION = False  # Desative se não tiver GPU


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, 'data', 'DataSet')