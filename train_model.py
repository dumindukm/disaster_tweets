import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Flatten, Dense, Embedding, LSTM, Dropout, Bidirectional, Input, Concatenate,GRU,SimpleRNN
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from common import tokenize_tweets, get_data_for_model, get_text_sequence


if __name__ == '__main__':
    pd.set_option('display.max_rows', 100000)
    # nltk.download('stopwords')
    # nltk.download('punkt')

    # tokenize words and remove stop words
    stop_words = set(stopwords.words('english'))

    # ds = pd.read_csv('data/train.csv')
    # ds = ds.fillna('0')

    # X = ds[['text']]
    # Y = ds[['target']]

    ds, X, Y = get_data_for_model('data/train.csv')

    # before tokenize let's remove urls from tweet
    # word_tokens = []

    # word_tokens = tokenize_tweets(ds)
    # filtered_sentence = [w for w in word_tokens if not w in list(stop_words)]


    # tokenizer = Tokenizer(num_words=20000)
    # tokenizer.fit_on_texts(filtered_sentence)

    # X_sequence = tokenizer.texts_to_sequences(filtered_sentence)
    # # print(X_sequence)
    # max_token_length = len(max(X_sequence, key=len))
    # print('max tokenlength', max_token_length)

    # vocab_size = len(tokenizer.word_index) + 1
    # print("vocab_size", vocab_size)


    # X_sequence = pad_sequences(X_sequence, padding='post',
    #                            maxlen=max_token_length)

    X_sequence, tokenizer, max_token_length, vocab_size = get_text_sequence(ds)

    # create word embeding
    embeddings_dictionary = dict()
    glove_file = open('data/glove.6B.100d.txt', encoding="utf8")

    for line in glove_file:
        records = line.split()
        word = records[0]
        vector_dimensions = np.asarray(records[1:], dtype='float32')
        embeddings_dictionary[word] = vector_dimensions
    glove_file.close()

    embeding_matrix = np.zeros((vocab_size, 100))

    for word, index in tokenizer.word_index.items():
        embeding = embeddings_dictionary.get(word)

        if(embeding is not None):
            embeding_matrix[index] = embeding


    # print(ds[["keyword_encoder"]].head())

    # df = pd.DataFrame(list(zip(X_sequence, le.transform(ds[["keyword"]]))),
    #                   columns=['text_sequence', 'keyword'])


    # split train and test data
    X_train_seq, X_test_seq, X_train_keyword, X_test_keyword, X_train_location, X_test_location,  y_train, y_test = train_test_split(
        X_sequence, ds[["keyword_encoder"]], ds[["location_encoder"]], Y, test_size=0.1, random_state=42)


    # X_train, X_test, y_train, y_test = train_test_split(
    #     df, Y, test_size=0.33, random_state=42)


    # Let's create sequnce model
    # model = Sequential()

    # embedding_layer = Embedding(vocab_size, 100, weights=[
    #                             embeding_matrix], input_length=max_token_length, trainable=False)
    # model.add(embedding_layer)
    # model.add(LSTM(15, return_sequences=True))
    # model.add(Dropout(.2))
    # model.add(Flatten())
    # model.add(Dense(2, activation='softmax'))

    inp_text_sequence = Input(shape=(max_token_length,))
    embedding_layer = Embedding(vocab_size, 100, weights=[
                                embeding_matrix], input_length=max_token_length, trainable=False)(inp_text_sequence)
    # embedding_layer = Embedding(vocab_size, 100, input_length=max_token_length, trainable=True)(inp_text_sequence)
    # lstm_text = SimpleRNN(120)(embedding_layer)
    lstm_text =  Bidirectional(LSTM(max_token_length))(embedding_layer)
    # lstm_text = LSTM(120)(embedding_layer)
    # lstm_text = LSTM(120)(embedding_layer)

    inp_keyword = Input(shape=(1,))


    inp_location = Input(shape=(1,))

    layer = Concatenate()([lstm_text, inp_location,inp_keyword])

    #*** new model



    #end new model

    # layer = lstm_text
    # layer = Dropout(.2)(layer)
    layer = Flatten()(layer)
    layer = Dense(256,activation='relu')(layer)
    # layer = Dropout(.2)(layer)
    # layer = Dense(128,activation='relu')(layer)
    output = Dense(2, activation='softmax')(layer)

    model = Model([inp_text_sequence,inp_location,inp_keyword], [output])

    modelCheckpoint = ModelCheckpoint('Checkpoints/tweet_disaster_model_5.h5',
                                    monitor='val_loss', verbose=0, save_best_only=True,  mode='auto', period=1)

    # model = load_model('Checkpoints/tweet_disaster_model_4.h5')
    model.compile(loss='sparse_categorical_crossentropy',
                optimizer=Adam(lr=0.0001), metrics=['accuracy'])


    model.fit([X_train_seq,X_train_location,X_train_keyword], y_train, epochs=50, batch_size=50,
            #   validation_split=.2,
            validation_data =([X_test_seq,X_test_location,X_test_keyword],y_test),
            callbacks=[modelCheckpoint])

    # model.save('tweet_disaster_model.h5')
    model = load_model('Checkpoints/tweet_disaster_model_5.h5')
    t_loss, t_accuracy = model.evaluate(
        [X_test_seq,X_test_location,X_test_keyword], y_test)

    print(t_loss, t_accuracy)
