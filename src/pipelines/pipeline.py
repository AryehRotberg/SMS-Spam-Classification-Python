import logging

import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation


# Logging Configuration
logging.basicConfig(filename='logs/pipeline.log',
                    format='[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

# Data Ingestion
data_ingestion = DataIngestion()
data_ingestion.preprocess_dataframe()
data_ingestion.split_data()
data_ingestion.to_csv(dataframe_processed_directory='data/processed',
                      train_directory='data/train',
                      test_directory='data/test')

logging.info('Created dataframe and splitted into train test.\n')

# Data Transformation
data_transformation = DataTransformation()
train_padded_sentences, test_padded_sentences = data_transformation.get_padded_sequences()
data_transformation.save_tokenizer(directory='outputs/tokenizers')

logging.info('Tokenized and padded sentences.\n')

# Model Training
model_trainer = ModelTrainer()
model_trainer.call()
model_trainer.compile()

train_labels = pd.read_csv('data/train/train_labels.csv').squeeze()
test_labels = pd.read_csv('data/test/test_labels.csv').squeeze()

history = model_trainer.fit(train_padded_sentences,
                            test_padded_sentences,
                            train_labels,
                            test_labels)

model_trainer.save('outputs/models')

train_labels = pd.read_csv('data/train/train_labels.csv').squeeze()
test_labels = pd.read_csv('data/test/test_labels.csv').squeeze()

# Model Evaluation - Train Set
model_evaluation = ModelEvaluation(train_padded_sentences, train_labels)
report = model_evaluation.get_classification_report()
conf_matrix = model_evaluation.get_confusion_matrix()

logging.info(f'Train Set -> Classification Report: \n{report}\n')
logging.info(f'Train Set -> Confusion Matrix: \n{conf_matrix}\n')

# Model Evaluation - Test Set
model_evaluation = ModelEvaluation(test_padded_sentences, test_labels)
report = model_evaluation.get_classification_report()
conf_matrix = model_evaluation.get_confusion_matrix()

logging.info(f'Test Set -> Classification Report: \n{report}\n')
logging.info(f'Test Set -> Confusion Matrix: \n{conf_matrix}\n')
