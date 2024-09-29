import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import numpy as np
from components.data_tranform import AmazonReviewProcessor

from utils import model_from_s3

class SentimentPredictor:
    def __init__(self, s3_model_bucket, s3_model_key, local_model_dir='artifacts/model/'):
        """
        Initializes the SentimentPredictor by downloading the model and tokenizer from S3.
        """
        self.local_model_dir = local_model_dir

        # Download and extract the model from S3
        if model_from_s3(s3_model_bucket, s3_model_key, self.local_model_dir):
            # Load the tokenizer and model from the local directory
            self.tokenizer = DistilBertTokenizer.from_pretrained(self.local_model_dir)
            self.model = TFDistilBertForSequenceClassification.from_pretrained(self.local_model_dir)
            self.label_map = {2: 'positive', 1: 'neutral', 0: 'negative'}

            # Initialize the text preprocessor
            self.preprocessor = AmazonReviewProcessor()
        else:
            raise ValueError("Failed to download and load the model from S3.")

    def preprocess(self, review):
        """
        Preprocesses and tokenizes the input text.
        """
        review = self.preprocessor.clean_text(review)
        review = review.lower()
        review = self.preprocessor.remove_stopwords(review)
        review = self.preprocessor.lemmatize_text(review)
        input_text = review.strip()

        # Handle None after preprocessing
        if input_text is None:
            raise ValueError("Input text could not be preprocessed. It might be non-English or empty.")

        # Tokenize the input text for DistilBERT
        tokens = self.tokenizer(
            text=input_text,
            padding=True,
            truncation=True,
            return_tensors="tf"
        )
        return tokens

    def predict(self, input_text):
        """
        Preprocesses the input text, performs the prediction, and returns the sentiment.
        """
        # Preprocess the input string
        tokens = self.preprocess(input_text)

        # Perform the prediction
        logits = self.model(tokens).logits
        predictions = tf.nn.softmax(logits, axis=-1)
        predicted_class = np.argmax(predictions, axis=1)[0]

        # Map the prediction to the sentiment label
        return self.label_map.get(predicted_class, "unknown")

