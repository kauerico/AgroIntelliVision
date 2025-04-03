# Configurações globais, como caminho do dataset, tamanho do batch, tamanho da imagem, número de épocas e número de classes
DATASET_PATH = "C:\Users\katys\OneDrive\Desktop\PROJETO IC\codeTest\AgroIntelliVision\data\raw\DataSet"
BATCH_SIZE = 64
IMG_SIZE = (256, 256)
EPOCHS = 40
NUM_CLASSES = 10

# Otimizações de performance
AUTOTUNE = tf.data.AUTOTUNE
MIXED_PRECISION = True