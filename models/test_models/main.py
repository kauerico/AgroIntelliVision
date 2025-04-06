import tensorflow as tf
import numpy as np
from deepchecks.vision import VisionData
from deepchecks.vision.suites import model_evaluation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Caminhos
MODEL_PATH = 'models/saved_models/1'
SAMPLE_PATH = 'data/sample_validation'
IMG_SIZE = (256, 256)
BATCH_SIZE = 32

# Carrega o modelo salvo
model = tf.keras.models.load_model(MODEL_PATH)

# Prepara o ImageDataGenerator para o sample
datagen = ImageDataGenerator(rescale=1./255)

sample_generator = datagen.flow_from_directory(
    SAMPLE_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Extrai imagens e labels
images, labels = [], []
for x_batch, y_batch in sample_generator:
    images.extend(x_batch)
    labels.extend(np.argmax(y_batch, axis=1))
    if len(images) >= sample_generator.samples:
        break

images = np.array(images)
labels = np.array(labels)

# Constrói o VisionData
ds = VisionData(images=images, labels=labels, task_type='classification')

# Roda a suite de validação
suite = model_evaluation()
result = suite.run(ds, model)

# Salva o relatório
result.save_as_html('validation_report.html')
