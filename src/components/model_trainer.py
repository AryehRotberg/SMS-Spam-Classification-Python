import os
import yaml

import numpy as np

import tensorflow as tf
import keras

from keras.layers import Embedding, Dense, GlobalAveragePooling1D

import pandas as pd
from sklearn.metrics import classification_report

np.random.seed(42)
tf.random.set_seed(42)


class ModelTrainer:
    def __init__(self):
        with open('config.yaml') as file:
            self.config = yaml.safe_load(file)
            self.config = self.config['model_training']
        
        self.model = keras.Sequential()

        self.embedding = Embedding(self.config['vocab_size'],
                                   self.config['embedding_dim'],
                                   input_length=self.config['max_text_length'])
        self.gap = GlobalAveragePooling1D()
        self.dense1 = Dense(24, activation='relu')
        self.dense2 = Dense(1, activation='sigmoid')
    
    def call(self):
        self.model.add(self.embedding)
        self.model.add(self.gap)
        self.model.add(self.dense1)
        self.model.add(self.dense2)
    
    def compile(self):
        self.model.compile(loss=self.config['loss'],
                           optimizer=self.config['optimizer'],
                           metrics=self.config['metrics'])
    
    def summary(self, logger):
        return self.model.summary(print_fn=logger.info)
    
    def fit(self, X_train, X_test, y_train, y_test):
        history = self.model.fit(X_train,
                                 y_train,
                                 epochs=self.config['num_epochs'],
                                 validation_data=(X_test, y_test),
                                 verbose=2)
        
        return history
    
    def predict(self, X):
        return self.model.predict(X)
    
    # def evaluate(self, X_test, y_test):
    #     return self.model.evaluate(X_test, y_test)
    
    def save(self, directory):
        self.model.save(os.path.join(directory, 'model.h5'))
    
    # def load(self, file_path):
    #     load_model(file_path)
    
    # def report(self, X_test, y_test):
    #     y_pred = self.predict(X_test) >= 0.5

    #     report = classification_report(y_test, y_pred, output_dict=True)
    #     report = pd.DataFrame(report).transpose()

    #     return report
