import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import numpy as np
from components.data_tranform import AmazonReviewProcessor
from utils import model_from_s3

class SentimentPredictor:
    def __init__(self, s3_model_bucket, s3_model_key, local_model_dir='artifacts/model'):
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

    def preprocess(self, input_text):
        """preprocesses and tokenize the input text."""
        
        input_text = self.preprocessor.preprocess_review(input_text)

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
    
if __name__ == "__main__":
    # Define your S3 configurations for the model
    S3_MODEL_BUCKET = 'your-model-s3-bucket-name'
    S3_MODEL_KEY = 'path/to/tokenizer_model.zip'  # Replace with your model ZIP key in S3
    
    # Initialize the SentimentPredictor
    predictor = SentimentPredictor(
        s3_model_bucket=S3_MODEL_BUCKET,
        s3_model_key=S3_MODEL_KEY,
    )
    
    # Example input
    input_review = "I absolutely love these earbuds! The sound quality is fantastic and they are very comfortable to wear."
    
    # Perform prediction
    sentiment = predictor.predict(input_review)
    print(f"Predicted Sentiment: {sentiment}")