import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from predict import SentimentPredictor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Fetch environment variables
SAGEMAKER_ENDPOINT_NAME = os.getenv('SAGEMAKER_ENDPOINT_NAME', 'sentiment-analysis-endpoint')
AWS_REGION = os.getenv('AWS_REGION')

# Initialize the SentimentPredictor
logger.info("Initializing SentimentPredictor...")
predictor = SentimentPredictor(
    endpoint_name=SAGEMAKER_ENDPOINT_NAME,
    region_name=AWS_REGION
)
logger.info("SentimentPredictor initialized successfully.")

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'review' not in data:
        logger.warning("No review text provided in the request.")
        return jsonify({'error': 'No review text provided.'}), 400
    
    input_text = data['review']
    logger.info(f"Received review for prediction: {input_text}")

    try:
        sentiment = predictor.predict(input_text)
        return jsonify({'sentiment': sentiment}), 200
    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        return jsonify({'error': 'An unexpected error occurred.'}), 500

if __name__ == "__main__":
    app.run(debug=True)
