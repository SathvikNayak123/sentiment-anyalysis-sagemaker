import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import numpy as np
from components.data_tranform import AmazonReviewProcessor

class SentimentPredictor:
    def __init__(self, model_dir='artifacts/model'):
        # Load the pre-trained model and tokenizer from the saved directory
        self.preprocessor = AmazonReviewProcessor()
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
        self.model = TFDistilBertForSequenceClassification.from_pretrained(model_dir)
        self.label_map = {2: 'positive', 1: 'neutral', 0: 'negative'}

    def preprocess(self, input_text):
        """preprocesses and tokenize the input text."""
        
        input_text = self.preprocessor.clean_text(input_text)
        input_text = self.preprocessor.remove_stopwords(input_text)
        input_text = self.preprocessor.lemmatize_text(input_text)
        input_text = input_text.strip()

        # Tokenize the input text for DistilBERT
        tokens = self.tokenizer(input_text, padding=True, truncation=True, return_tensors="tf")
        return tokens

    def predict(self, input_text):
        """Preprocesses the input text, performs the prediction, and returns the sentiment."""
        # Preprocess the input string
        tokens = self.preprocess(input_text)

        # Perform the prediction
        logits = self.model(tokens).logits
        predictions = tf.nn.softmax(logits, axis=-1)
        predicted_class = np.argmax(predictions, axis=1)[0]

        # Map the prediction to the sentiment label
        return self.label_map[predicted_class]