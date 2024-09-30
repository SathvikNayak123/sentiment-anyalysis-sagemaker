import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from transformers import pipeline
from datasets import Dataset
from src.components.utils import import_from_s3, upload_to_s3

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

class DataProcessor:
    def __init__(self, s3_raw_bucket, s3_raw_key, s3_clean_bucket, s3_clean_key):
        self.s3_raw_bucket = s3_raw_bucket
        self.s3_raw_key = s3_raw_key
        self.s3_clean_bucket = s3_clean_bucket
        self.s3_clean_key = s3_clean_key
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        self.candidate_labels = ['positive', 'negative', 'neutral']
        DetectorFactory.seed = 0
        self.class_weights = {'positive': 1.0,'negative': 1.0,'neutral': 10.0}
        self.amazon_df = None

    def import_data_from_s3(self):
        """
        Import the dataset from S3.
        """
        self.amazon_df = import_from_s3(self.s3_raw_bucket, self.s3_raw_key)
        if self.amazon_df is None:
            raise ValueError("Failed to import DataFrame from S3.")

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
        if not isinstance(review, str):
            return None
        if self.detect_language(review) != 'en':
            return None
        review = self.clean_text(review)
        review = review.lower()
        review = self.remove_stopwords(review)
        review = self.lemmatize_text(review)
        return review.strip() if len(review.strip()) > 0 else None


    def apply_preprocessing(self):
        # Apply preprocessing to the Reviews column
        self.amazon_df['Cleaned_Reviews'] = self.amazon_df['Review'].apply(self.preprocess_review)
        # Remove rows where Cleaned_Reviews is None (indicating non-English or empty reviews)
        initial_count = len(self.amazon_df)
        self.amazon_df.dropna(subset=['Cleaned_Reviews'], inplace=True)
        final_count = len(self.amazon_df)
        print(f"Preprocessing complete. Dropped {initial_count - final_count} non-English or empty reviews.")

    def classify_reviews(self):
        # Convert DataFrame to Hugging Face Dataset for efficient batch processing
        dataset = Dataset.from_pandas(self.amazon_df[['Cleaned_Reviews']])

        # Function to classify each review using Zero-Shot Classification
        def classify_batch(batch):
            result = self.classifier(batch['Cleaned_Reviews'], self.candidate_labels, multi_label=False)
            adjusted_pseudo_labels = []
            for res in result:
                # Extract labels and scores
                labels = res['labels']
                scores = res['scores']

                # Adjust scores with class weights
                adjusted_scores = [score * self.class_weights[label] for label, score in zip(labels, scores)]

                # Assign the label with the highest adjusted score
                pseudo_label = labels[adjusted_scores.index(max(adjusted_scores))]
                adjusted_pseudo_labels.append(pseudo_label)

            return {'Pseudo_Labels': adjusted_pseudo_labels}

        # Apply the classification in batches using the map function
        dataset = dataset.map(classify_batch, batched=True, batch_size=16)  # Adjust batch size based on your GPU memory
        # Convert back to DataFrame
        classified_df = dataset.to_pandas()
        return classified_df[['Cleaned_Reviews', 'Pseudo_Labels']]

    def process_reviews(self):
        """
        Full processing pipeline: preprocess and classify reviews.
        """
        print("Starting preprocessing of reviews...")
        self.apply_preprocessing()
        print("Preprocessing done. Starting classification of reviews...")
        classified_df = self.classify_reviews()
        print("Classification done.")
        return classified_df

    def save_cleaned_data_to_s3(self, classified_df):
        """
        Uploads the classified DataFrame to S3 as a CSV.
        """
        return upload_to_s3(classified_df, self.s3_clean_bucket, self.s3_clean_key)
