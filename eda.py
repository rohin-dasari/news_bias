import matplotlib.pyplot as plt
from util import build_dataset
from transformers import BertTokenizer


tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
ds, _ = build_dataset(
        'datasets/titles.csv',
        tokenizer,
        val_size=0.0)

#hist, bins = ds['labels']
plt.hist(ds['labels'], bins='auto')
plt.show()

