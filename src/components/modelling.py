import os
import shutil
import pandas as pd
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
from sklearn.model_selection import StratifiedShuffleSplit
import nltk
from utils import (
    import_from_s3,
    model_to_s3,
    model_from_s3
)

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

class DataPrep:
    def __init__(self, s3_input_bucket, s3_input_key, s3_model_bucket, s3_model_key):
        """
        Initializes the DataPrep class with S3 configurations.
        """
        self.s3_input_bucket = s3_input_bucket
        self.s3_input_key = s3_input_key
        self.s3_model_bucket = s3_model_bucket
        self.s3_model_key = s3_model_key
        self.label_map = {'positive': 2, 'neutral': 1, 'negative': 0}
        self.tokenizer = None

    def import_data(self):
        """
        Imports the dataset from S3.
        """
        print("Importing data from S3...")
        df = import_from_s3(self.s3_input_bucket, self.s3_input_key)
        if df is None:
            raise ValueError("Failed to import DataFrame from S3.")
        print("Data imported successfully.")
        return df

    def preprocess_data(self, df):
        """
        Maps labels and splits the data into training, validation, and testing sets.
        """
        print("Preprocessing data...")
        df = df.copy()
        df['Pseudo_Labels'] = df['Pseudo_Labels'].map(self.label_map)
        
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in sss.split(df, df['Pseudo_Labels']):
            train_df = df.iloc[train_index]
            test_df = df.iloc[test_index]
        print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
        
        sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, val_index in sss_val.split(train_df, train_df['Pseudo_Labels']):
            train_split_df = train_df.iloc[train_index]
            val_df = train_df.iloc[val_index]
        print(f"Training size: {len(train_split_df)}, Validation size: {len(val_df)}")
        
        return train_split_df, val_df, test_df

    def download_tokenizer(self):
        """
        Downloads the tokenizer/model from S3.
        """
        print("Downloading tokenizer/model from HuggingFace...")
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    def tokenize_data(self, texts, padding=True, truncation=True, return_tensors="tf"):
        """
        Tokenizes the input texts using the downloaded tokenizer.
        """
        return self.tokenizer(
            texts,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
            clean_up_tokenization_spaces=False
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


class ModelTrainer:
    def __init__(self, s3_output_model_bucket, s3_output_model_key, num_labels=3, learning_rate=5e-5):
        """
        Initializes the ModelTrainer with model configurations and S3 output details.
        """
        self.model_dir = 'artifacts/model/'
        self.s3_output_model_bucket = s3_output_model_bucket
        self.s3_output_model_key = s3_output_model_key
        self.num_labels = num_labels
        self.learning_rate = learning_rate
        self.model = None
        self.early_stopping = None

    def initialize_model(self):
        """
        Loads the DistilBERT model for sequence classification.
        """
        print("Initializing the model...")
        self.model = TFDistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            num_labels=self.num_labels
        )
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        self.early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=2,
            restore_best_weights=True
        )
        print("Model initialized successfully.")

    def train(self, train_dataset, val_dataset, epochs=10):
        """
        Trains the model using the training and validation datasets.
        """
        print("Starting model training...")
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=[self.early_stopping]
        )
        print("Model training completed.")
        return history

    def evaluate(self, test_dataset):
        """
        Evaluates the model on the testing dataset.
        """
        print("Evaluating the model...")
        eval_loss, eval_acc = self.model.evaluate(test_dataset)
        print(f"Test Accuracy: {eval_acc}")
        return eval_acc

    def save_model(self):
        """
        Saves the trained model to a specified directory.
        """
        self.model.save_pretrained(self.model_dir)
        print(f"Model saved to '{self.model_dir}'.")

    def upload_model_to_s3(self):
        """
        Uploads the trained model directory to S3.
        """
        print("Uploading trained model to S3...")
        if model_to_s3(self.model_dir, self.s3_output_model_bucket, self.s3_output_model_key):
            print("Trained model successfully uploaded to S3.")
        else:
            print("Failed to upload trained model to S3.")