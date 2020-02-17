import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from common import tokenize_tweets, get_data_for_model, get_text_sequence


ds, X, Y = get_data_for_model('data/test.csv')
X_sequence, tokenizer, max_token_length, vocab_size = get_text_sequence(ds)

model = load_model('Checkpoints/tweet_disaster_model.h5')
results = model.predict([X_sequence, ds[["keyword_encoder"]]])

test_results = pd.DataFrame(
    {'id': ds['id'], 'target': np.argmax(results, axis=1)})
test_results.to_csv('data/results_1.csv', index=False)

print(np.argmax(results, axis=1))
