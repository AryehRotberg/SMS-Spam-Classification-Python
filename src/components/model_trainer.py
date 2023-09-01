import numpy as np
import pandas as pd

import tensorflow as tf
import keras

from keras.models import load_model
from keras.layers import Embedding, Dense, GlobalAveragePooling1D

from sklearn.metrics import classification_report


np.random.seed(42)
tf.random.set_seed(42)


class Model:
    def __init__(self, input_length):
        self.vocab_size = 1000
        self.embedding_dim = 16
        self.loss = 'binary_crossentropy'
        self.optimizer = 'adam'
        self.metrics = ['accuracy']
        self.num_epochs = 30

        self.model = keras.Sequential()
        self.model.add(Embedding(self.vocab_size,
                                 self.embedding_dim,
                                 input_length=input_length))
        self.model.add(GlobalAveragePooling1D())
        self.model.add(Dense(24, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))

        self.model.compile(loss=self.loss,
                           optimizer=self.optimizer,
                           metrics=self.metrics)
    
    def fit(self, X_train, y_train, X_test, y_test):
        history = self.model.fit(X_train,
                                 y_train,
                                 epochs=self.num_epochs,
                                 validation_data=(X_test, y_test),
                                 verbose=2)
        
        return history
    
    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)
    
    def save(self, file_path):
        self.model.save(file_path)
    
    def load(self, file_path):
        load_model(file_path)
    
    def report(self, X_test, y_test):
        y_pred = self.predict(X_test) >= 0.5

        report = classification_report(y_test, y_pred, output_dict=True)
        report = pd.DataFrame(report).transpose()

        return report
