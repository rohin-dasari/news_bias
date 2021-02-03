import os
import pandas as pd
import numpy as np
import logging
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

def get_callbacks():

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        os.path.join('checkpoints', 'tf_model.h5'),
        monitor='val_loss',
        save_best_only=True
        )
    earlystopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, mode='max')

    return [tensorboard_callback, checkpoint_callback, earlystopping_callback]


if __name__ == '__main__':
    train_ds, val_ds, encoder = build_dataset('datasets/blurbs.csv', tokenizer, 0.2)
    train_ds = train_ds.shuffle(100).batch(64)
    val_ds = val_ds.shuffle(100).batch(64)
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    loss = tf.keras.losses.MeanSquaredError()

    model.compile(
        optimizer=optimizer,
        loss=loss,
        )

    log_dir = get_logdir('logs/')

    logging.info('writing all logs to {}'.format(log_dir))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    history = model.fit(
        train_ds,
        epochs=1,
        callbacks=get_callbacks(),
        validation_data=val_ds
        )

    logging.info('wrote all logs to {}'.format(log_dir))

    model_dir = get_logdir('models', prefix='m')
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    model.save_pretrained(model_dir)
    logging.info('saved model to {}'.format(model_dir))

