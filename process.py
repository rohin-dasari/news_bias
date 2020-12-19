import os
import re
import multiprocessing as mp
from tqdm import tqdm
import pandas as pd
from dataclasses import dataclass, field
from typing import Iterable

pattern = re.compile(r'[\a\b\f\n\r\t]')


def getText(path):
    words = []
    with open(path, 'r') as f:
        [words.extend(line) for line in f.readlines()]
    words_str = ''.join(words)
    return pattern.sub(' ', words_str).strip()


def checkAscii(text):
    try:
        text.encode('ascii')
        return True
    except UnicodeEncodeError:
        return False


def getTitle(text):
    return text.split('.')[0]

def getTitleAndBlurb(text, max_sentences, max_chars):
    res = ''
    sentences = text.split('.')
    for i, sentence in enumerate(sentences):
        res += sentence
        if i > max_sentences or len(res) > max_chars:
            break

    return res


def splitText(text, max_chars, max_sentences=None):
    sentences = text.split('.')

    if max_sentences is None:
        max_sentences = len(sentences)

    blurb = ''
    res = []

    for i, sentence in enumerate(sentences[:max_sentences]):
        if len(blurb) + len(sentence) > max_chars:
            res.append(blurb.strip())
            blurb = ''
        blurb += sentence
    if len(res) == 0:
        res.append(sentences[0])
    return res


def getScore(path):
    return float(path.split('_')[-1])


def extract_text_for_experiments(
        path,
        min_characters=200,
        min_characters_per_blurb=300,
        min_characters_per_split=400,
        blurb_limit=4,
        title_limit=2):

    text = getText(path)

    if not checkAscii(text):
        return None, 'not ascii encodable'
    if len(text) < min_characters:
        return None, 'too short'
    
    # get title
    title = ''.join(splitText(
        text,
        min_characters,
        max_sentences=title_limit))

    # get title+blurb; split blurb within desired no. of chars
    title_and_blurb = ''.join(splitText(
        text,
        min_characters_per_blurb,
        max_sentences=blurb_limit))
    # get full article; split blurb within desired no. of chars

    full_text_split = splitText(text, min_characters_per_split)

    return title, title_and_blurb, full_text_split, getScore(path), path


class Data:
    def __init__(self):
        self.data = []
        self.labels = []

    def add(self, text, label):
        if isinstance(text, Iterable) and not isinstance(text, str):
            self.data.extend(text)
            self.labels.extend(label)
            return

        self.data.append(text)
        self.labels.append(label)

    def __dict__(self):
        return {'data': self.data,
                'labels': self.labels}

    def to_csv(self, path):
        dict_repr = self.__dict__()
        pd.DataFrame(dict_repr).to_csv(path)



if __name__ == '__main__':
    files = [os.path.join('./data', f) for f in os.listdir('./data')]
    count = 0
    non_ascii_count = 0
    short_count = 0
    valid_articles = Data()
    titles = Data()
    blurbs = Data()
    full_articles = Data()
    #full_articles = {'data': [], 'scores': []}
    with tqdm(total=len(files)) as pbar:
        for res in map(
                extract_text_for_experiments,
                files):
            if None in res:
                if 'too short' in res[1]:
                    short_count += 1
                else:
                    non_ascii_count += 1
                continue
            title, blurb, full_text, score, path = res
            titles.add(title, score)
            #titles['data'].append(title)
            #titles['scores'].append(score)
            blurbs.add(blurb, score)
            #blurbs['data'].append(blurb)
            #blurbs['scores'].append(score)
            full_articles.add(full_text, [score]*len(full_text))
            #full_articles['data'].extend(full_text)
            #full_articles['scores'].extend([score]*len(full_text))
            valid_articles.add(path, score)
            pbar.update()
        print('total number of invalid articles %d'
              % (non_ascii_count + short_count))
        print('%d articles are not ascii encodable' % non_ascii_count)
        print('%d articles are too short'
              '(likely a failed pull by the web scraper)' % short_count)
        print('total number of valid articles %d'
              % (len(files)-non_ascii_count))

    if not os.path.isdir('./datasets'):
        os.makedirs('./datasets')
    valid_articles.to_csv('datasets/valid_articles.csv')
    titles.to_csv('datasets/titles.csv')
    blurbs.to_csv('datasets/blurbs.csv')
    full_articles.to_csv('datasets/full_articles.csv')
