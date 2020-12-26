import pandas as pd
import numpy as np
import trafilatura as tr
import json, os
import multiprocessing as mp
from tqdm import tqdm
#from transformers import BertTokenizer, TFBertForSequenceClassification


#tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
#model = TFBertForSequenceClassification.from_pretrained('bert-base-cased', return_dict=True)


data_file = 'newsArticlesWithLabels.tsv'
data = pd.read_csv(data_file, delimiter='\t')


# get text from urls
# convert discrete labels to continuous encoding
# preprocess text from url
# - remove backslashes
# - remove new line characters
# - get title 
# - BERT can only accept 512 tokens at a time


def get_text_from_url(url):
    download = tr.fetch_url(url) 
    text = tr.extract(download, json_output=True)
    if text is None: 
        return {'title': None, 'text': None}
    article = json.loads(text)
    return article
    #return article['title'] + '. ' + article['text']


def write_to_file(data, label, filename):
    data_dir = os.path.join(os.getcwd(), 'data')
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    with open(os.path.join(data_dir, filename+'_'+str(label)), 'w+') as f:
        f.write(data)

def get_article(df):
    score_reference_num = [-1, -0.5, 0, 0.5, 1]
    score_reference = ['VeryPositive',
            'SomewhatPositive',
            'Neutral',
            'SomewhatNegative',
            'VeryNegative']

    with tqdm(total=df.shape[0]) as pbar:
        for i, row in df.iterrows():
            try:
                article = get_text_from_url(row.url)
                
                if article['title'] is not None and article['text'] is not None:
                    dem_bias, rep_bias = row['democrat.vote'], row['republican.vote']
                    dem_score = score_reference_num[score_reference.index(dem_bias)]
                    rep_score = score_reference_num[::-1][score_reference.index(rep_bias)]
                    score = (dem_score + rep_score)/2
                    write_to_file(
                            article['title'] + '. ' + article['text'],
                            score,
                            article['title'].replace(' ', '').replace('/', ''))
            except ValueError:
                pass
            pbar.update(1)

def pull_articles(df):
    num_processes = mp.cpu_count()-1
    pool = mp.Pool(processes=num_processes)
    pool.map(get_article, np.array_split(df, num_processes))

if __name__ == '__main__':
    pull_articles(data)


