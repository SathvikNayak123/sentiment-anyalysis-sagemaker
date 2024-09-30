# components/predict.py

import boto3
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentPredictor:
    def __init__(self, endpoint_name, region_name='us-east-1'):
        """
        Initializes the SentimentPredictor by setting up the SageMaker runtime client.
        
        Args:
            endpoint_name (str): The name of the SageMaker endpoint.
            region_name (str, optional): AWS region. Defaults to 'us-east-1'.
        """
        self.endpoint_name = endpoint_name
        self.region_name = region_name
        self.sagemaker_runtime = boto3.client('sagemaker-runtime', region_name=self.region_name)

    def preprocess(self, input_text):
        """
        Preprocesses the input text as required by the model.
        
        Args:
            input_text (str): The raw input text.
        
        Returns:
            str: The preprocessed text in JSON format.
        """
        # Implement any necessary preprocessing here
        # For simplicity, assuming the model expects a JSON payload with 'review' key
        payload = json.dumps({"review": input_text})
        return payload

    def predict(self, input_text):
        """
        Sends the preprocessed text to the SageMaker endpoint and returns the prediction.
        
        Args:
            input_text (str): The raw input text.
        
        Returns:
            str: Predicted sentiment ('positive', 'neutral', 'negative').
        """
        payload = self.preprocess(input_text)
        
        try:
            response = self.sagemaker_runtime.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType='application/json',
                Body=payload
            )
            
            result = json.loads(response['Body'].read().decode())
            sentiment = result.get('sentiment', 'unknown')
            logger.info(f"Predicted Sentiment: {sentiment}")
            return sentiment
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return "error"
