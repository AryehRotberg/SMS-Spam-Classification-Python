import streamlit as st

import yaml

import pickle

from keras_preprocessing.sequence import pad_sequences

from keras.models import load_model


def get_configs():
    with open('main.yaml') as file:
        config = yaml.safe_load(file)

    config_tokenization = config['Tokenization']
    config_model_selection = config['Model_Selection']

    return config_tokenization, config_model_selection


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
                                maxlen=171,
                                padding=config_tokenization['padding_type'],
                                truncating=config_tokenization['trunc_type'])
    
    return padded_text


def configure_site():
    st.title('SMS Spam Classification')

    text_input = st.text_input('Insert text')
    st.text("")
    button = st.button('Predict')

    return text_input, button


config_tokenization, config_model_selection = get_configs()
model = load_classifier()
tokenizer = load_tokenizer()

text_input, button = configure_site()

if button:
    padded_text = preprocess_text(text_input)

    prediction = model.predict(padded_text)[0][0]
    
    if prediction >= .5:
        st.write('Spam! Probability: ', prediction)
    else:
        st.write('Not a spam. Probability: ', prediction)
