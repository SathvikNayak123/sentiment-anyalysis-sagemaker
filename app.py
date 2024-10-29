from flask import Flask, render_template, request
from components.predict import SentimentPredictor

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    input_text = request.form['inputText']

    predictor = SentimentPredictor(input_text)
    prediction = predictor.predict()
    
    return render_template('index.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)

