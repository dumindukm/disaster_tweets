from spellchecker import SpellChecker
import re
import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from multiprocessing import Pool
from sklearn.preprocessing import LabelEncoder
import string
from nltk.stem.porter import PorterStemmer

spell = SpellChecker()

porter = PorterStemmer() 

# https://machinelearningmastery.com/clean-text-machine-learning-python/
def remove_html(text):
    html = re.compile(r'<.*?>')
    return html.sub(r'', text)


def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'', text)


def remove_emoloji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def remove_punct(text):
    table = str.maketrans('', '', string.punctuation)
    return text.translate(table)




def correct_spellings(text):
    corrected_text = []
    misspelled_words = spell.unknown(text.split())
    for word in text.split():
        if word in misspelled_words:
            corrected_text.append(spell.correction(word))
        else:
            corrected_text.append(word)
    print(corrected_text)
    return " ".join(corrected_text)


def text_clean(text):

    # text = text.lower()
    text = correct_spellings(text)
    # text = remove_URL(text)
    # text = remove_html(text)
    # text = remove_emoloji(text)
    # text = remove_punct(text)
    # text = correct_spellings(text)
    return text


def tokenize_tweets(ds):
    word_tokens = []

    ds['text'] = ds['text'].apply(lambda x: text_clean(x))
    stop_words = list(stopwords.words('english'))
    for sentence in ds['text']:
        # sentence = re.sub(r'((http(s)?(\:\/\/))+(www\.)?([\w\-\.\/])*(\.[a-zA-Z]{2,3}\/?))[^\s\b\n|]*[^.,;:\?\!\@\^\$ -]',
        #                   '', sentence, flags=re.MULTILINE)
        # print(sentence)
        # print('************', word_tokenize(sentence))
        tk =  [ w for w in word_tokenize(sentence) if (w not in stop_words) and  w.isalpha() and len(w) > 2]
        # print('##############', tk)
        word_tokens.append(tk)
    return word_tokens

def find_mention_count(ds):

    print('Target {}'. format( len( ds[ds.text.str.count('@')> 0  ] ) ))

def clean_data(ds):
  ds['text'] = ds['text'].apply(lambda x: text_clean(x))
  return ds

def parallelize_dataframe(df, func, n_cores=4):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

def get_data_for_model(data_file_path):
    ds = pd.read_csv(data_file_path)
    ds['location'].fillna(value='unk', inplace=True)
    ds['keyword'].fillna(value='k_unk', inplace=True)
    ds = ds.fillna('0')

    ds = parallelize_dataframe(ds,clean_data)
    ds.to_csv('test_clean.csv',index=False)

    X = ds[['text']] 
    # find_mention_count(ds )
    ds["new_text"] = ds['text'] + ds['keyword'] + ds['location']
    Y = None
    if 'target' in ds:
        Y = ds[['target']]

    le = LabelEncoder()
    le.fit(ds[["keyword"]])
    ds["keyword_encoder"] = le.transform(ds[["keyword"]])

    # le = LabelEncoder()
    le.fit(ds[["location"]])
    ds["location_encoder"] = le.transform(ds[["location"]])

    return ds, X, Y

  
def get_text_sequence(ds):
    # nltk.download('stopwords')
    # nltk.download('punkt')

    # tokenize words and remove stop words
    stop_words = set(stopwords.words('english'))
    word_tokens = []

    word_tokens = tokenize_tweets(ds)


    filtered_sentence = [w for w in word_tokens if not w in list(stop_words)]
    
    

    tokenizer = Tokenizer(num_words=20000)
    tokenizer.fit_on_texts(filtered_sentence)

    X_sequence = tokenizer.texts_to_sequences(filtered_sentence)

    max_token_length = 72  # len(max(X_sequence, key=len))
    print('max tokenlength', max_token_length)

    vocab_size = len(tokenizer.word_index) + 1
    print("vocab_size", vocab_size)

    X_sequence = pad_sequences(X_sequence, padding='post',
                               maxlen=max_token_length)

    return X_sequence, tokenizer, max_token_length, vocab_size
