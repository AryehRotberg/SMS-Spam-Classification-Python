import streamlit as st

import pickle

from keras_preprocessing.sequence import pad_sequences

from keras.models import load_model


max_length = 171
padding_type = 'post'
trunc_type = 'post'

def load_classifier():
    model = load_model('outputs/model.h5')
    return model

def load_tokenizer():
    with open('outputs/tokenizer.pickle', 'rb') as file:
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

def configure_site():
    st.title('SMS Spam Classification')

    text_input = st.text_input('Insert text')
    st.text("")
    button = st.button('Predict')

    return text_input, button


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
