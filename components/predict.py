from data_transform import DataProcessor, Split_Tokenize_Data
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk



class SentimentPredictor:
    def __init__(self, input):
        self.input=input
        self.preprocessor = DataProcessor()
        self.tokenize = Split_Tokenize_Data()
        self.model = TFDistilBertForSequenceClassification.from_pretrained('artifacts/model')

    def preprocess(self):
        self.input = self.preprocessor.preprocess_review(self.input)
        self.tokenize.download_tokenizer()
        self.input = self.tokenize.tokenize_data(self.input)

    def predict(self):
        self.preprocess()
        
        logits = self.model(**self.input).logits
        prediction = np.argmax(logits, axis=1).numpy()[0]

        predicted_label = self.tokenize.label_map[prediction]
        return predicted_label

    
