import os
import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from transformers import TFBertForSequenceClassification
from util import get_logdir, build_dataset
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import OneHotEncoder


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=1
        )

            
class ContinuousPrecision(tf.keras.metrics.Metric):
    
    def __init__(self, name='Continuous Precision', **kwargs):
        super(ContinuousPrecision, self).__init__(name=name, **kwargs)
        #self._name = name
        self.precision = tf.keras.metrics.Precision()

    def update_state(self, y_true, y_pred, sample_weight=None):
        def clip(arr, clip_value):
            return tf.cast(arr > clip_value, arr.dtype) * arr
        y_true = clip(y_true, 0)
        y_pred = clip(y_pred, 0)
        self.precision.update_state(y_true, y_pred)

    def result(self):
        return self.precision.result()

    def reset_states(self):
        self.precision.reset_states()

train_ds, val_ds, encoder = build_dataset('datasets/titles.csv', tokenizer, 0.2)
train_ds = train_ds.shuffle(100).batch(64)
val_ds = val_ds.shuffle(100).batch(64)
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
loss = tf.keras.losses.MeanSquaredError()

model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=[ContinuousPrecision()])

log_dir = get_logdir('logs/')

print('writing all logs to {}'.format(log_dir))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
history = model.fit(
    train_ds,
    epochs=1,
    callbacks=[tensorboard_callback],
    validation_data=val_ds
    )

print('wrote all logs to {}'.format(log_dir))

model_dir = get_logdir('models', prefix='m')
if not os.path.isdir(model_dir):
    os.makedirs(model_dir)
model.save_pretrained(model_dir)
print('saved model to {}'.format(model_dir))

