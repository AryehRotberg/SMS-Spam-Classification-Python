import yaml

from fastapi import FastAPI
from pydantic import BaseModel

import pickle

from keras_preprocessing.sequence import pad_sequences

from keras.models import load_model


max_length = 171
padding_type = 'post'
trunc_type = 'post'

with open('config.yaml') as file:
    config = yaml.safe_load(file)
    config = config['data_transformation']

def load_classifier():
    model = load_model('outputs/models/model.h5')
    return model

def load_tokenizer():
    with open('outputs/tokenizer.pickle', 'rb') as file:
        tokenizer = pickle.load(file)
    
    return tokenizer

def preprocess_text(text):
    text = [text]
    sequence = tokenizer.texts_to_sequences(text)
    padded_text = pad_sequences(sequence,
                                maxlen=config['max_text_length'],
                                padding=config['padding_type'],
                                truncating=config['trunc_type'])
    
    return padded_text


class Input(BaseModel):
    text_input: str


model = load_classifier()
tokenizer = load_tokenizer()

app = FastAPI()

@app.get('/')
def home():
    return 'Health Check'

@app.post('/predict')
def predict(input: Input):
    padded_text = preprocess_text(input.text_input)

    prediction = model.predict(padded_text)[0][0]
    
    if prediction >= .5:
        return {'class': 'Spam',
                'confidence': float(prediction)}
    else:
        return {'class': 'Not Spam',
                'confidence': float(prediction)}
