import pandas as pd

from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, f1_score


class ModelEvaluation:
    def __init__(self, sentences, labels):
        self.model = load_model('outputs/models/model.h5')

        self.sentences = sentences
        self.labels = labels

        self.pred = self.model.predict(self.sentences) >= 0.5
    
    def evaluate(self):
        return self.model.evaluate(self.sentences, self.labels)

    def get_classification_report(self):
        return pd.DataFrame(classification_report(self.labels, self.pred, output_dict=True)).transpose()
    
    def get_confusion_matrix(self):
        return confusion_matrix(self.labels, self.pred)
    
    def get_f1_score(self):
        return f1_score(self.labels, self.pred)
