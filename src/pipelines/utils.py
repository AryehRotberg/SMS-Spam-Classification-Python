import pandas as pd

from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences


class utils:
    def __init__(self):
        self.oov_token = '<OOV>'
        self.padding_type = 'post'
        self.trunc_type = 'post'
        self.vocab_size = 1000
    
    def preprocess_dataframe(self, df: pd.DataFrame):
        df = df.rename(columns={'v1': 'label', 'v2': 'text'})
        df.label = df.label == 'spam'
        df['text_length'] = df.text.apply(lambda text : len(text.split()))

        return df
    
    def get_padded_sequences(self, df: pd.DataFrame, train_sentences: pd.DataFrame, test_sentences: pd.DataFrame):
        tokenizer = Tokenizer(num_words=self.vocab_size,
                              oov_token=self.oov_token)
        
        tokenizer.fit_on_texts(train_sentences)

        max_text_length = df.text_length.max()

        train_sequences = tokenizer.texts_to_sequences(train_sentences)
        train_padded = pad_sequences(train_sequences,
                                     maxlen=max_text_length,
                                     padding=self.padding_type,
                                     truncating=self.trunc_type)
        
        test_sequences = tokenizer.texts_to_sequences(test_sentences)
        test_padded = pad_sequences(test_sequences,
                                    maxlen=max_text_length,
                                    padding=self.padding_type,
                                    truncating=self.trunc_type)
        
        return tokenizer, train_padded, test_padded
