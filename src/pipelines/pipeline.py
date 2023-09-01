import pickle

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from utils import utils
from src.components.model_trainer import Model


np.random.seed(42)

# Data Ingestion and Transformation
utils = utils()

spam_df = pd.read_csv('data/spam_dataset.csv', encoding='latin-1', usecols=['v1', 'v2'])
spam_df = utils.preprocess_dataframe(spam_df)

train_sentences, test_sentences, train_labels, test_labels = train_test_split(spam_df.text, spam_df.label, test_size=0.3)
tokenizer, train_padded, test_padded = utils.get_padded_sequences(spam_df, train_sentences, test_sentences)

# Model Training
model = Model(input_length=spam_df.text_length.max())
history = model.fit(train_padded, train_labels, test_padded, test_labels)

# Model Evaluation
evaluation = model.evaluate(test_padded, test_labels)
report = model.report(test_padded, test_labels)

print(report)
print(evaluation)

# Model Saving
with open('outputs/tokenizer.pickle', 'wb') as file:
    pickle.dump(tokenizer, file, protocol=pickle.HIGHEST_PROTOCOL)

model.save('outputs/models/model.h5')
