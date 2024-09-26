import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from transformers import pipeline
from datasets import Dataset

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

class AmazonReviewProcessor:
    def __init__(self):
        self.amazon_df = pd.read_csv('artifacts/amazon_data.csv')
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        self.candidate_labels = ['positive', 'negative', 'neutral']
        DetectorFactory.seed = 0  # Consistency in language detection

    def detect_language(self, text):
        try:
            return detect(text)
        except LangDetectException:
            return 'unknown'

    def clean_text(self, review):
        # Remove URLs, HTML tags, special characters, and digits
        review = re.sub(r'http\S+|www\S+|https\S+', '', review, flags=re.MULTILINE)
        review = re.sub(r'<.*?>', '', review)
        review = re.sub(r'\d+', '', review)
        review = re.sub(r'[^A-Za-z\s]+', '', review)  # Keep only alphabets and spaces
        return review

    def remove_stopwords(self, review):
        return ' '.join([word for word in review.split() if word not in self.stop_words])

    def lemmatize_text(self, review):
        return ' '.join([self.lemmatizer.lemmatize(word) for word in review.split()])

    def preprocess_review(self, review):
        if isinstance(review, str):  # Ensure the review is a valid string
            # Detect language, and only process if it's English
            if self.detect_language(review) != 'en':
                return None
            # Apply cleaning, stopword removal, and lemmatization
            review = self.clean_text(review)
            review = review.lower()
            review = self.remove_stopwords(review)
            review = self.lemmatize_text(review)
            return review.strip() if len(review.strip()) > 0 else None
        return None

    def apply_preprocessing(self):
        # Apply preprocessing to the Reviews column
        self.amazon_df['Cleaned_Reviews'] = self.amazon_df['Review'].apply(self.preprocess_review)
        # Remove rows where Cleaned_Reviews is None (indicating non-English or empty reviews)
        self.amazon_df.dropna(subset=['Cleaned_Reviews'], inplace=True)

    def classify_reviews(self):
        # Convert DataFrame to Hugging Face Dataset for efficient batch processing
        dataset = Dataset.from_pandas(self.amazon_df[['Cleaned_Reviews']])

        # Function to classify each review using Zero-Shot Classification
        def classify_batch(batch):
            result = self.classifier(batch['Cleaned_Reviews'], self.candidate_labels)
            # Get the label with the highest score for each review
            return {'Pseudo_Labels': [res['labels'][0] for res in result]}

        # Apply the classification in batches using the map function
        dataset = dataset.map(classify_batch, batched=True, batch_size=16)  # Adjust batch size based on your GPU memory
        # Convert back to DataFrame
        classified_df = dataset.to_pandas()
        return classified_df[['Cleaned_Reviews', 'Pseudo_Labels']]

    def save_cleaned_data(self, output_file):
        classified_df = self.classify_reviews()
        classified_df.to_csv(output_file, index=False)

# Usage example
if __name__ == '__main__':
    processor = AmazonReviewProcessor()
    processor.apply_preprocessing()
    processor.save_cleaned_data('artifacts/data_cleaned.csv')