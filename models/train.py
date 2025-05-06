from tensorflow import keras
from config import settings

def get_optimizer(train_steps=None):
    # Learning rate com warmup
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-3,
        decay_steps=train_steps * 20 if train_steps else 1000,
        alpha=1e-5
    )
    
    return keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=1e-4,
        global_clipnorm=1.0
    )

def compile_model(model, optimizer):
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc', multi_label=True),
            keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_accuracy')
        ]
    )
    return model