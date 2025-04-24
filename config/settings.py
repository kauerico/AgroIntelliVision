# Configurações globais
DATASET_PATH = "/content/drive/MyDrive/seu_caminho_para_dataset"
BATCH_SIZE = 32  # Reduzi para melhor uso de memória
IMG_SIZE = (256, 256)  # Mantenha consistente com seu modelo
EPOCHS = 50  # Aumentei um pouco
NUM_CLASSES = 8  # Corrigi para bater com suas classes reais

# Otimizações de performance
AUTOTUNE = tf.data.AUTOTUNE
MIXED_PRECISION = True