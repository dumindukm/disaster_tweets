import re
import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


from sklearn.preprocessing import LabelEncoder


def tokenize_tweets(ds):
    word_tokens = []
    for sentence in ds['text']:
        sentence = re.sub(r'((http(s)?(\:\/\/))+(www\.)?([\w\-\.\/])*(\.[a-zA-Z]{2,3}\/?))[^\s\b\n|]*[^.,;:\?\!\@\^\$ -]',
                          '', sentence, flags=re.MULTILINE)
        # print(sentence)
        word_tokens.append(word_tokenize(sentence))
    return word_tokens


def get_data_for_model(data_file_path):
    ds = pd.read_csv(data_file_path)
    ds = ds.fillna('0')

    X = ds[['text']]
    Y = None
    if 'target' in ds:
        ds[['target']]

    le = LabelEncoder()
    le.fit(ds[["keyword"]])
    ds["keyword_encoder"] = le.transform(ds[["keyword"]])

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
