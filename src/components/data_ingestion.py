import os
import yaml

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

np.random.seed(42)


class DataIngestion:
    def __init__(self):
        self.spam_df = pd.read_csv('data/raw/spam_dataset.csv', encoding='latin-1', usecols=['v1', 'v2'])

        with open('config.yaml') as file:
            self.config = yaml.safe_load(file)
            self.config = self.config['data_ingestion']
    
    def preprocess_dataframe(self):
        self.spam_df = self.spam_df.rename(columns={'v1': 'label', 'v2': 'text'})
        self.spam_df.label = self.spam_df.label == 'spam'
        self.spam_df['text_length'] = self.spam_df.text.apply(lambda text : len(text.split()))
    
    def split_data(self):
        self.train_sentences, self.test_sentences, self.train_labels, self.test_labels = train_test_split(self.spam_df.text,
                                                                                                          self.spam_df.label,
                                                                                                          test_size=self.config['test_size'])
    
    def to_csv(self, dataframe_processed_directory, train_directory, test_directory):
        self.spam_df.to_csv(os.path.join(dataframe_processed_directory, 'spam_df.csv'), index=False)

        self.train_sentences.to_csv(os.path.join(train_directory, 'train_sentences.csv'), index=False)
        self.train_labels.to_csv(os.path.join(train_directory, 'train_labels.csv'), index=False)

        self.test_sentences.to_csv(os.path.join(test_directory, 'test_sentences.csv'), index=False)
        self.test_labels.to_csv(os.path.join(test_directory, 'test_labels.csv'), index=False)
