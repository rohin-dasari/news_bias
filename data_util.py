import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf



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
            # in the future avoid reading in the entire csv file into a dataframe
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


