import os
from dotenv import load_dotenv
from flask import Flask, request, render_template
from components.predict import SentimentPredictor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

ROLE = os.getenv('SM_ROLE')
ENDPOINT_NAME = os.getenv('ENDPOINT_NAME')
AWS_REGION = os.getenv('AWS_REGION')

logger.info("Initializing SentimentPredictor...")
predictor = SentimentPredictor(
    endpoint_name=ENDPOINT_NAME,
    region_name=AWS_REGION
)
logger.info("SentimentPredictor initialized successfully.")

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['inputText']
    prediction = predictor.predict(input_text)
    return render_template('index.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
