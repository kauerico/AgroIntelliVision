import sys
import os
import tensorflow as tf
import numpy as np
from deepchecks.vision import VisionData
from deepchecks.vision.suites import model_evaluation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Argumentos da linha de comando (modelo e imagens)
model_path = sys.argv[1] if len(sys.argv) > 1 else 'models/saved_models/1'
sample_path = sys.argv[2] if len(sys.argv) > 2 else 'data/sample_validation'

# Parâmetros
IMG_SIZE = (256, 256)
BATCH_SIZE = 32

print(f'📦 Carregando modelo de: {model_path}')
model = tf.keras.models.load_model(model_path)

print(f'🖼️ Carregando imagens de: {sample_path}')
datagen = ImageDataGenerator(rescale=1./255)

sample_generator = datagen.flow_from_directory(
    sample_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Extrai todas as imagens e labels
images, labels = [], []
for x_batch, y_batch in sample_generator:
    images.extend(x_batch)
    labels.extend(np.argmax(y_batch, axis=1))
    if len(images) >= sample_generator.samples:
        break

images = np.array(images)
labels = np.array(labels)

print(f'🔍 Gerando dataset para validação...')
ds = VisionData(images=images, labels=labels, task_type='classification')

print(f'✅ Rodando suite de validação Deepchecks...')
suite = model_evaluation()
result = suite.run(ds, model)

# Salva o relatório
report_path = 'validation_report.html'
result.save_as_html(report_path)
print(f'📄 Relatório salvo em: {report_path}')