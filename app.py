from flask import Flask, render_template, request
from components.predict import SentimentPredictor
import os
from dotenv import load_dotenv

app = Flask(__name__)

load_dotenv()
S3_MODEL_BUCKET = os.getenv('S3_MODEL_BUCKET')
S3_MODEL_KEY = os.getenv('S3_MODEL_KEY')

predictor = SentimentPredictor(
    s3_model_bucket=S3_MODEL_BUCKET,
    s3_model_key=S3_MODEL_KEY
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['inputText']
    prediction = predictor.predict(input_text)
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
