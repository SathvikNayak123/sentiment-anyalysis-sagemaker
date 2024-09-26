from flask import Flask, render_template, request
from components.predict import SentimentPredictor

app = Flask(__name__)

predictor = SentimentPredictor()

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
