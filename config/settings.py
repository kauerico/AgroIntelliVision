import tensorflow as tf

# Configurações globais
# Altere para:
DATASET_PATH = r"C:\Users\katys\OneDrive\Documentos\GitHub\AgroIntelliVision\data\DataSet"  # Note o 'r' antes da string # Ajuste conforme necessário
BATCH_SIZE = 32  # Reduzi para melhor uso de memória
IMG_SIZE = (256, 256)  # Mantenha consistente com seu modelo
EPOCHS = 50  # Aumentei um pouco
NUM_CLASSES = 8  # Corrigi para bater com suas classes reais
# Limitação de tamanho de arquivo


# Otimizações de performance
AUTOTUNE = tf.data.AUTOTUNE
MIXED_PRECISION = True