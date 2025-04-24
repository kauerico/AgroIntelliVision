from tensorflow import keras
from config import settings

def get_optimizer(steps_per_epoch):
    lr_schedule = keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=1e-3,  # Aumentei a taxa inicial
        first_decay_steps=steps_per_epoch * 5,
        t_mul=1.5,
        m_mul=0.85  # Ajuste fino
    )
    return keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=1e-5  # Reduzi o weight decay
    )

def compile_model(model, optimizer):
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc', curve='ROC'),  # Adicionei curve ROC
            keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_accuracy')  # Nova métrica
        ]
    )
    return model