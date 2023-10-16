import os
import yaml

import pickle

import numpy as np
import pandas as pd

from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

np.random.seed(42)


class DataTransformation:
    def __init__(self):
        self.spam_df = pd.read_csv('data/processed/spam_df.csv')
        self.train_sentences = pd.read_csv('data/train/train_sentences.csv').squeeze()
        self.test_sentences = pd.read_csv('data/test/test_sentences.csv').squeeze()

        with open('config.yaml') as file:
            self.config = yaml.safe_load(file)
            self.config = self.config['data_transformation']
    
    def get_padded_sequences(self):
        self.tokenizer = Tokenizer(num_words=self.config['vocab_size'],
                                   oov_token=self.config['oov_token'])
        
        self.tokenizer.fit_on_texts(self.train_sentences)

        max_text_length = self.spam_df.text_length.max()

        train_sequences = self.tokenizer.texts_to_sequences(self.train_sentences)
        train_padded = pad_sequences(train_sequences,
                                     maxlen=max_text_length,
                                     padding=self.config['padding_type'],
                                     truncating=self.config['trunc_type'])
        
        test_sequences = self.tokenizer.texts_to_sequences(self.test_sentences)
        test_padded = pad_sequences(test_sequences,
                                    maxlen=max_text_length,
                                    padding=self.config['padding_type'],
                                    truncating=self.config['trunc_type'])
        
        return train_padded, test_padded
    
    def save_tokenizer(self, directory):
        with open(os.path.join(directory, 'tokenizer.pickle'), 'wb') as file:
            pickle.dump(self.tokenizer, file, protocol=pickle.HIGHEST_PROTOCOL)
