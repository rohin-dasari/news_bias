import os
import numpy as np
import pandas as pd
from copy import copy
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import OneHotEncoder
from transformers import BertTokenizer
from transformers import TFBertForSequenceClassification
import tensorflow as tf


def manual_slice(data, labels, slice_obj):
    """
    slice a batch encoding and return the slice
    """
    sliced_data = {}
    sliced_labels = labels[slice_obj]
    for (k, v) in data.items():
        sliced_data[k] = v[slice_obj]
    return sliced_data, sliced_labels

def load_model_and_tokenizer(model_path):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = TFBertForSequenceClassification.from_pretrained(model_path)
    return model, tokenizer


def process_input(samples, tokenizer):
    tokenized = tokenizer(
        samples,
        max_length=200,
        truncation=True,
        padding=True)
    for k, v in tokenized.items():
        tokenized[k] = np.array(v, ndmin=2)
    return tokenized


def build_dataset(path, tokenizer, val_size, return_type='tf'):
    

    data_df = pd.read_csv(path)
    data_df['data'] = data_df['data'].apply(str)
    data_df['labels'] = data_df['labels'].apply(float)
    label_values = np.array(data_df['labels'].values)[:, None]
    data = np.array(data_df['data'].values)[:, None]

    def get_dataset(X, y):
        if X is None or y is None:
            return None
        tokenized = tokenizer(
            list(np.squeeze(X)),
            max_length=200,
            truncation=True,
            padding=True)

        def gen():
            #column_names = ['index', 'data', 'labels']
            #with open(path, 'r') as f:
            #    for i, line in enumerate(f.readlines()):
            #        if i == 0: # dont read the column names
            #            continue

            #        pass
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
                tf.float32),
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
    else:
        train_X, train_y, test_X, test_y = data, label_values, None, None

    if return_type == 'tf': # training code will expect the tf dataset format 
        return get_dataset(train_X, train_y), get_dataset(test_X, test_y)
    elif return_type == 'dict': # allowing for a dict type is useful for debugging
        return {'data': train_X, 'labels': train_y}, {'data': test_X, 'labels': test_y}
    else:
        raise ValueError('invalid return type')


def get_logdir(base_dir, prefix=''):
    try:
        dirs = os.listdir(base_dir)
    except FileNotFoundError:
        dirs = []

    valid_dir = []
    for d in dirs:
        try:
            valid_dir.append(int(d.replace(prefix, '')))
        except ValueError:
            continue
    valid_dir_sorted = sorted(valid_dir)
    new_dir = valid_dir_sorted[-1]+1 if len(valid_dir_sorted) > 0 else 0
    return os.path.join(base_dir, prefix+str(new_dir))


class ContinuousPrecision(tf.keras.metrics.Metric):
    
    def __init__(self, name='Continuous Precision', **kwargs):
        super(ContinuousPrecision, self).__init__(name=name, **kwargs)
        #self._name = name
        self.precision = tf.keras.metrics.Precision()

    def update_state(self, y_true, y_pred, sample_weight=None):
        def clip(arr, clip_value):
            return tf.cast(arr > clip_value, y_pred.dtype) * tf.cast(arr, y_pred.dtype)
        # if y_pred is within x% of y_true, count as true
        y_pred_discrete = []
        clearance = tf.cast(0.3, y_pred.dtype)
        y_true = clip(y_true, 0)
        y_true_upper_bound = y_true + clearance
        y_true_lower_bound = y_true - clearance
        less_than_upper_bound = tf.cast(y_pred < y_true_upper_bound, y_pred.dtype)*y_pred
        greater_than_lower_bound = tf.cast(
            less_than_upper_bound > y_true_lower_bound,
            y_pred.dtype)
        zero_idx = tf.where(greater_than_lower_bound == 0)
        try:
            tf.tensor_scatter_nd_update(y_pred, zero_idx, tf.cast([200]*zero_idx.shape[0], y_pred.dtype))
        except:
            pass
        y_pred = clip(y_pred, 0)
        self.precision.update_state(y_true, y_pred)

    def result(self):
        return self.precision.result()

    def reset_states(self):
        self.precision.reset_states()
