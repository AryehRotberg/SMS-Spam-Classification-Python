from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

import pickle

from keras_preprocessing.sequence import pad_sequences

from keras.models import load_model


max_length = 171
padding_type = 'post'
trunc_type = 'post'


def load_classifier():
    model = load_model('Outputs/model.h5')
    return model


def load_tokenizer():
    with open('Outputs/tokenizer.pickle', 'rb') as file:
        tokenizer = pickle.load(file)
    
    return tokenizer


def preprocess_text(text):
    text = [text]
    sequence = tokenizer.texts_to_sequences(text)
    padded_text = pad_sequences(sequence,
                                maxlen=max_length,
                                padding=padding_type,
                                truncating=trunc_type)
    
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
        return f'Spam! Probability: {prediction}'
    else:
        return f'Not a spam. Probability: {prediction}'
