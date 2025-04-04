import tensorflow as tf

# Configurações globais, como caminho do dataset, tamanho do batch, tamanho da imagem, número de épocas e número de classes
DATASET_PATH = "data/raw/DataSet"
BATCH_SIZE = 64
IMG_SIZE = (256, 256)
EPOCHS = 20
NUM_CLASSES = 16

# Otimizações de performance
AUTOTUNE = tf.data.AUTOTUNE
MIXED_PRECISION = True