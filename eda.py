import matplotlib.pyplot as plt
from transformers import BertTokenizer
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pandas as pd


tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

def build_dataset(path, tokenizer, val_size):
    
    label_mapping = {
            -1: 0,
            -0.5: 0.16,
            -0.25: 0.32, 
            0: 0.48,
            0.25: 0.64,
            0.5: 0.80,
            1: 1, 
            }

    data_df = pd.read_csv(path)
    data_df['data'] = data_df['data'].apply(str)
    data_df['labels'] = data_df['labels'].apply(float).apply(lambda l: label_mapping[l])
    label_values = np.array(data_df['labels'].values)[:, None]
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(label_values) 
    data = np.array(data_df['data'].values)[:, None]


    if val_size > 0:
        train_X, test_X, train_y, test_y = train_test_split(
                                                            data,
                                                            label_values,
                                                            test_size=val_size,
                                                            stratify=label_values)
        train_y = enc.transform(train_y)
        oversample = RandomOverSampler(sampling_strategy='minority') 
        for i in range(10):
            train_X, train_y = oversample.fit_sample(
                train_X,
                train_y)
        return {'data': train_X, 'labels': enc.inverse_transform(train_y)}, {'data': test_X, 'labels': test_y}
    
    return data_df, None

ds, _ = build_dataset(
        'datasets/titles.csv',
        tokenizer,
        val_size=0.2)
#hist, bins = ds['labels']
plt.hist(ds['labels'], bins='auto')
plt.show()

