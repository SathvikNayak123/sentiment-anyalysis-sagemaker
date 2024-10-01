import subprocess
import sys
import tensorflow as tf
from transformers import DistilBertTokenizer
from sklearn.model_selection import StratifiedShuffleSplit
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from transformers import pipeline
from datasets import Dataset
import argparse
import logging
import re
import boto3
from io import StringIO
import pandas as pd
from botocore.exceptions import NoCredentialsError, ClientError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

class Utils:
    def __init__(self):
        self.s3_client=boto3.client('s3')

    def get_data_s3(self, s3_bucket, s3_key):

        try:
            # Get the CSV object from S3
            response = self.s3_client.get_object(Bucket=s3_bucket, Key=s3_key)
            # Read the CSV data into a pandas DataFrame
            csv_content = response['Body'].read().decode('utf-8')
            df = pd.read_csv(StringIO(csv_content))
            return df
        except ClientError as e:
            print(f"Failed to retrieve data from S3: {e}")
            return None

    def put_data_s3(self, df, s3_bucket, s3_key):

        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        try:
            self.s3_client.put_object(Bucket=s3_bucket, Key=s3_key, Body=csv_buffer.getvalue())
            print(f"Data successfully uploaded to s3://{s3_bucket}/{s3_key}")
            return True
        except ClientError as e:
            print(f"Failed to upload to S3: {e}")
            return False


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

        self.utils = Utils()
        self.amazon_df = None

    def import_data_from_s3(self):
        self.amazon_df = self.utils.get_data_s3(self.s3_raw_bucket, self.s3_raw_key)

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
        self.amazon_df['Cleaned_Reviews'] = self.amazon_df['Review'].apply(self.preprocess_review)
        initial_count = len(self.amazon_df)
        self.amazon_df.dropna(subset=['Cleaned_Reviews'], inplace=True)
        final_count = len(self.amazon_df)
        print(f"Preprocessing complete. Dropped {initial_count - final_count} non-English or empty reviews.")

    def classify_reviews(self):
        dataset = Dataset.from_pandas(self.amazon_df[['Cleaned_Reviews']])

        def classify_batch(batch):
            result = self.classifier(batch['Cleaned_Reviews'], self.candidate_labels, multi_label=False)
            adjusted_pseudo_labels = []
            for res in result:
                labels = res['labels']
                scores = res['scores']
                adjusted_scores = [score * self.class_weights[label] for label, score in zip(labels, scores)]

                pseudo_label = labels[adjusted_scores.index(max(adjusted_scores))]
                adjusted_pseudo_labels.append(pseudo_label)
            return {'Pseudo_Labels': adjusted_pseudo_labels}

        # Apply the classification in batches using the map function
        dataset = dataset.map(classify_batch, batched=True, batch_size=16)  
        # Convert back to DataFrame
        classified_df = dataset.to_pandas()
        return classified_df[['Cleaned_Reviews', 'Pseudo_Labels']]

    def ProcessReviews(self):
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
        return self.utils.put_data_s3(classified_df, self.s3_clean_bucket, self.s3_clean_key)
    
class Split_Tokenize_Data:
    def __init__(self, s3_clean_bucket, s3_clean_key, s3_data_bucket, s3_train_key, s3_val_key, s3_test_key):

        self.s3_clean_bucket = s3_clean_bucket
        self.s3_clean_key = s3_clean_key
        self.s3_data_bucket = s3_data_bucket
        self.s3_train_key = s3_train_key
        self.s3_val_key = s3_val_key
        self.s3_test_key = s3_test_key

        self.label_map = {'positive': 2, 'neutral': 1, 'negative': 0}
        self.tokenizer = None

        self.utils = Utils()
        self.clean_df = None

    def import_data_from_s3(self):
        self.clean_df = self.utils.get_data_s3(self.s3_clean_bucket, self.s3_clean_key)

    def preprocess_data(self):
        """
        Maps labels and splits the data into training, validation, and testing sets.
        """
        print("Preprocessing data...")
        self.clean_df['Pseudo_Labels'] = self.clean_df['Pseudo_Labels'].map(self.label_map)
        
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in sss.split(self.clean_df, self.clean_df['Pseudo_Labels']):
            train_df = self.clean_df.iloc[train_index]
            test_df = self.clean_df.iloc[test_index]
        print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
        
        sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, val_index in sss_val.split(train_df, train_df['Pseudo_Labels']):
            train_split_df = train_df.iloc[train_index]
            val_df = train_df.iloc[val_index]
        print(f"Training size: {len(train_split_df)}, Validation size: {len(val_df)}")
        
        return train_split_df, val_df, test_df

    def download_tokenizer(self):
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    def tokenize_data(self, texts, padding=True, truncation=True, return_tensors="tf"):
        return self.tokenizer(
            texts,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors
        )

    def create_datasets(self, train_df, val_df, test_df, batch_size=16):
        """
        Creates TensorFlow datasets from the DataFrames.
        """
        print("Creating TensorFlow datasets...")
        self.download_tokenizer()
        train_encodings = self.tokenize_data(train_df['Cleaned_Reviews'].tolist())
        val_encodings = self.tokenize_data(val_df['Cleaned_Reviews'].tolist())
        test_encodings = self.tokenize_data(test_df['Cleaned_Reviews'].tolist())
        
        train_dataset = tf.data.Dataset.from_tensor_slices((
            dict(train_encodings),
            train_df['Pseudo_Labels'].values
        )).shuffle(10000).batch(batch_size)
        
        val_dataset = tf.data.Dataset.from_tensor_slices((
            dict(val_encodings),
            val_df['Pseudo_Labels'].values
        )).batch(batch_size)
        
        test_dataset = tf.data.Dataset.from_tensor_slices((
            dict(test_encodings),
            test_df['Pseudo_Labels'].values
        )).batch(batch_size)
        
        print("Datasets created successfully.")
        return train_dataset, val_dataset, test_dataset
    
    def save_all_datasets(self, train_dataset, val_dataset, test_dataset):
        train_dataset.save('artifacts/train_data')
        val_dataset.save('artifacts/val_data')
        test_dataset.save('artifacts/test_data')

        print("-----------Saved Train, Val and Test datasets to artifacts--------\n")
    
def main(args):
    
    S3_RAW_BUCKET = args.s3_raw_bucket
    S3_RAW_KEY = args.s3_raw_key
    S3_CLEAN_BUCKET = args.s3_clean_bucket
    S3_CLEAN_KEY = args.s3_clean_key

    logger.info(f"---Data Preprocess Starting---")

    processor = DataProcessor(S3_RAW_BUCKET, S3_RAW_KEY, S3_CLEAN_BUCKET, S3_CLEAN_KEY)
    processor.import_data_from_s3()
    classified_df = processor.ProcessReviews()
    processor.save_cleaned_data_to_s3(classified_df)

    logger.info(f"---Splitting & Tokenizing Data---")

    transform = Split_Tokenize_Data(S3_CLEAN_BUCKET, S3_CLEAN_KEY)
    transform.import_data_from_s3()
    train_df, val_df, test_df = transform.preprocess_data()
    train_dataset, val_dataset, test_dataset = transform.create_datasets(train_df, val_df, test_df)
    transform.save_all_datasets(train_dataset, val_dataset, test_dataset)

    logger.info(f"---Data Transformation Complete---")

if __name__ == "__main__":

    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

    parser = argparse.ArgumentParser(description="SageMaker Preprocessing Script")

    # Input S3 paths
    parser.add_argument(
        "--s3-raw-bucket",
        type=str,
        required=True,
        help="S3 bucket containing raw data"
    )
    parser.add_argument(
        "--s3-raw-key",
        type=str,
        required=True,
        help="S3 key (prefix) for raw data"
    )

    # Output S3 paths
    parser.add_argument(
        "--s3-clean-bucket",
        type=str,
        required=True,
        help="S3 bucket to store cleaned data"
    )
    parser.add_argument(
        "--s3-clean-key",
        type=str,
        required=True,
        help="S3 key (prefix) for cleaned data"
    )

    args = parser.parse_args()
    main(args)