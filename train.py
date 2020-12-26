import os
import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from transformers import TFBertForSequenceClassification
from util import get_logdir
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import OneHotEncoder




tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=1
        )

def random_oversampler(X, y, label_type):
    pass


def build_dataset(path, tokenizer, val_size):

    data_df = pd.read_csv(path)
    data_df['data'] = data_df['data'].apply(str)
    data_df['labels'] = data_df['labels'].apply(float)
    label_values = np.array(data_df['labels'].values)[:, None]
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(label_values) 
    data = np.array(data_df['data'].values)[:, None]

    def get_dataset(X, y):
        tokenized = tokenizer(
            list(np.squeeze(X)),
            max_length=200,
            truncation=True,
            padding=True)

        def gen():
            for idx, label in enumerate(y):
                yield (
                    {'input_ids': tokenized['input_ids'][idx],
                     'attention_mask': tokenized['attention_mask'][idx],
                     'token_type_ids': tokenized['token_type_ids'][idx]
                        },
                     label)

        return tf.data.Dataset.from_generator(gen,
                ({
                    'input_ids': tf.int32,
                    'attention_mask': tf.int32,
                    'token_type_ids': tf.int32,
                },
                tf.int32),
                ({
                    'input_ids': tf.TensorShape([None]),
                    'attention_mask': tf.TensorShape([None]),
                    'token_type_ids': tf.TensorShape([None])
                },
                tf.TensorShape([None]))
                )

    if val_size > 0:
        train_X, test_X, train_y, test_y = train_test_split(
                                                            data,
                                                            label_values,
                                                            test_size=val_size,
                                                            stratify=label_values)
        oversample = RandomOverSampler(sampling_strategy='minority') 
        #print(train_df.shape)
        train_X_over, train_y_over = oversample.fit_sample(
            train_X,
            enc.fit_transform(train_y))
        #train_df_over = pd.DataFrame({'data': train_df_over, 'one_hot_labels': train_labels_over})
        return get_dataset(train_X_over, enc.inverse_transform(train_y_over)), get_dataset(test_X, test_y)

    return get_dataset(data_df), None

#def xentropy(labels, logits):
    

train_ds, val_ds = build_dataset('datasets/titles.csv', tokenizer, 0.2)
train_ds = train_ds.shuffle(100).batch(64)
val_ds = val_ds.shuffle(100).batch(64)
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
loss = tf.keras.losses.MeanSquaredError()

model.compile(
    optimizer=optimizer,
    loss=loss)
log_dir = get_logdir('logs/')

print('writing all logs to {}'.format(log_dir))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
history = model.fit(
    train_ds,
    epochs=2,
    callbacks=[tensorboard_callback],
    validation_data=val_ds
    )

#print('wrote all logs to {}'.format(log_dir))
#
#model_dir = 'models/m1'
#if not os.path.isdir(model_dir):
#    os.makedirs(model_dir)
#model.save_pretrained(model_dir)
#print('saved model to {}'.format(model_dir))

