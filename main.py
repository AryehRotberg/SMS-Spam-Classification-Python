import pickle

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

from Model import Model


np.random.seed(42)

dataset_location = 'Data/spam_dataset.csv'
oov_token = '<OOV>'
vocab_size = 1000
padding_type = 'post'
trunc_type = 'post'


def preprocess_dataframe(df):
    df.rename(columns={'v1': 'Label', 'v2': 'Text'}, inplace=True)
    df.Label = spam_df.Label == 'spam'
    df['Text_Length'] = spam_df.Text.apply(lambda text : len(text.split()))


def get_padded_inputs(train_sentences, test_sentences):
    tokenizer = Tokenizer(num_words=vocab_size,
                          oov_token=oov_token)

    tokenizer.fit_on_texts(train_sentences)

    train_sequences = tokenizer.texts_to_sequences(train_sentences)
    train_padded = pad_sequences(train_sequences,
                                 maxlen=spam_df.Text_Length.max(),
                                 padding=padding_type,
                                 truncating=trunc_type)

    test_sequences = tokenizer.texts_to_sequences(test_sentences)
    test_padded = pad_sequences(test_sequences,
                                maxlen=spam_df.Text_Length.max(),
                                padding=padding_type,
                                truncating=trunc_type)
    
    return tokenizer, train_padded, test_padded


spam_df = pd.read_csv(dataset_location, encoding='latin-1', usecols=['v1', 'v2'])
preprocess_dataframe(spam_df)

train_sentences, test_sentences, train_labels, test_labels = train_test_split(spam_df.Text, spam_df.Label, test_size=0.3)
tokenizer, train_padded, test_padded = get_padded_inputs(train_sentences, test_sentences)

model = Model(input_length=spam_df.Text_Length.max())
history = model.fit(train_padded, train_labels, test_padded, test_labels)

evaluation = model.evaluate(test_padded, test_labels)
report = model.report(test_padded, test_labels)

print(report)
print(evaluation)

with open('Outputs/tokenizer.pickle', 'wb') as file:
    pickle.dump(tokenizer, file, protocol=pickle.HIGHEST_PROTOCOL)

model.save('Outputs/model.h5')
