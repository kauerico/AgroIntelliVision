from tensorflow import keras
from config import settings

def get_optimizer(steps_per_epoch):
    lr_schedule = keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=1e-4,
        first_decay_steps=steps_per_epoch * 5,
        t_mul=1.5,
        m_mul=0.9
    )
    return keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=1e-4
    )

def compile_model(model, optimizer):
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    return model