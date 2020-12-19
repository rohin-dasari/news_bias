import os
import numpy as np
import pandas as pd
from copy import copy
from sklearn.model_selection import train_test_split


def manual_slice(data, labels, slice_obj):
    """
    slice a batch encoding and return the slice
    """
    sliced_data = {}
    sliced_labels = labels[slice_obj]
    for (k, v) in data.items():
        sliced_data[k] = v[slice_obj]
    return sliced_data, sliced_labels


def build_dataset(path, tokenizer, val_size):
    # tokenize text using bert tokenizer
    # split data into train and test set
    # store data as tf dataset
    # return dataset objects
    def batch_tokenize(df):
        # tokenize text
        # return tf dataset object
        tokenized = tokenizer(
                list(df['data'].values),
                max_length=200,
                truncation=True,
                padding=True)

        for k, v in tokenized.items():
            tokenized[k] = np.array(v)
        
        return {'tokens': tokenized, 'labels': df['labels'].values}

    data_df = pd.read_csv(path)
    data_df['data'] = data_df['data'].apply(str)
    data_df['labels'] = data_df['labels'].apply(float)
    if val_size > 0:
        train_df, test_df = train_test_split(data_df, test_size=val_size)
        return batch_tokenize(train_df), batch_tokenize(test_df)
    return batch_tokenize(data_df), {'tokens': [], 'labels': []}


def shuffle_and_batch(data, labels, batch_size):
    data, labels = copy(data), copy(labels)
    # pad the data and labels to get even splits on the batch size
    while len(labels) % batch_size != 0:
        idx = np.random.randint(0, len(labels))
        labels = np.append(labels, labels[idx])
        values = {}
        for k, v in data.items():
            values[k] = v[idx]
        for k, v in values.items():
            data[k] = np.append(data[k], v[None, :], axis=0)

    # shuffle the data
    shuffled_idx = np.random.permutation(len(labels))
    shuffled_labels = labels[shuffled_idx]
    shuffled_data = {}
    for k, v in data.items():
        #print(v.shape)
        shuffled_data[k] = v[shuffled_idx, :]
    
    # batch the data together
    n_batches = int(len(shuffled_labels) / batch_size)
    batched_labels = np.split(
            shuffled_labels,
            n_batches)

    batched_data = []

    for i in range(n_batches):
        batched_data.append({})

    for k, v in shuffled_data.items():
        batches = np.split(v, n_batches)
        for i, b in enumerate(batches):
            if k not in batched_data:
                batched_data[i][k] = []
            batched_data[i][k].append(b)

    for d in batched_data:
        for k, v in d.items():
            d[k] = np.squeeze(np.array(v))

    return batched_data, batched_labels


def get_logdir(base_dir):
    try:
        dirs = os.listdir(base_dir)
    except:
        dirs = []
    valid_dir = []
    for d in dirs:
        try:
            valid_dir.append(int(d))
        except ValueError:
            continue
    valid_dir_sorted = sorted(valid_dir)
    #print(valid_dir_sorted)
    new_dir = valid_dir_sorted[-1]+1 if len(valid_dir_sorted) > 0 else 0
    return os.path.join(base_dir, str(new_dir))

