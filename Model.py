import yaml

import numpy as np
import pandas as pd

import tensorflow as tf
import keras

from keras.models import load_model
from keras.layers import Embedding, Dense, GlobalAveragePooling1D

from sklearn.metrics import classification_report


np.random.seed(42)
tf.random.set_seed(42)

def get_configs():
    with open('main.yaml') as file:
        config = yaml.safe_load(file)

    config_tokenization = config['Tokenization']
    config_model_training = config['Model_Training']

    return config_tokenization, config_model_training


class Model:
    def __init__(self, input_length):
        self.config_tokenization, self.config_model_training = get_configs()

        self.model = keras.Sequential()
        self.model.add(Embedding(self.config_tokenization['vocab_size'],
                            self.config_model_training['embedding_dim'],
                            input_length=input_length))
        self.model.add(GlobalAveragePooling1D())
        self.model.add(Dense(24, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))

        self.model.compile(loss=self.config_model_training['loss'],
                           optimizer=self.config_model_training['optimizer'],
                           metrics=self.config_model_training['metrics'])
    
    def fit(self, X_train, y_train, X_test, y_test):
        history = self.model.fit(
            X_train,
            y_train,
            epochs=self.config_model_training['num_epochs'],
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
    
    def save_as_tflite(self, file_path):
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.post_training_quantize = True
        tflite_model = converter.convert()

        with open(file_path, 'wb') as file:
            file.write(tflite_model)
