# experiment.py
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

class DataProcessor:
    def __init__(self, df):
        self.df = df.copy()
        self.label_map = {'positive': 2, 'neutral': 1, 'negative': 0}
        self.df['Pseudo_Labels'] = self.df['Pseudo_Labels'].map(self.label_map)
    
    def split_data(self):
        """
        Splits the data into training, validation, and testing sets using stratified sampling.
        """
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in sss.split(self.df, self.df['Pseudo_Labels']):
            self.train_df = self.df.iloc[train_index]
            self.test_df = self.df.iloc[test_index]
        print(f"Train size: {len(self.train_df)}, Test size: {len(self.test_df)}")
        
        sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, val_index in sss_val.split(self.train_df, self.train_df['Pseudo_Labels']):
            self.train_split_df = self.train_df.iloc[train_index]
            self.val_df = self.train_df.iloc[val_index]
        print(f"Training size: {len(self.train_split_df)}, Validation size: {len(self.val_df)}")

        return self.train_split_df, self.val_df, self.test_df

class Tokenizer:
    def __init__(self, tokenizer_path):
        """
        Initializes the Tokenizer by loading it from a specified path.
        """
        self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)
    
    def tokenize_data(self, data):
        
        return self.tokenizer(
            data,
            padding=True,
            truncation=True,
            return_tensors="tf",
            clean_up_tokenization_spaces=False
        )

class DataLoader:
    def __init__(self, train_df, val_df, test_df, tokenizer):

        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.tokenizer = tokenizer

    def create_datasets(self, batch_size=16):
        """
        Creates TensorFlow datasets from the DataFrames.
        """
        train_encodings = self.tokenizer.tokenize_data(self.train_df['Cleaned_Reviews'].tolist())
        val_encodings = self.tokenizer.tokenize_data(self.val_df['Cleaned_Reviews'].tolist())
        test_encodings = self.tokenizer.tokenize_data(self.test_df['Cleaned_Reviews'].tolist())

        train_dataset = tf.data.Dataset.from_tensor_slices((
            dict(train_encodings),
            self.train_df['Pseudo_Labels'].values
        )).shuffle(10000).batch(batch_size)

        val_dataset = tf.data.Dataset.from_tensor_slices((
            dict(val_encodings),
            self.val_df['Pseudo_Labels'].values
        )).batch(batch_size)

        test_dataset = tf.data.Dataset.from_tensor_slices((
            dict(test_encodings),
            self.test_df['Pseudo_Labels'].values
        )).batch(batch_size)

        return train_dataset, val_dataset, test_dataset

class ModelTrainer:
    def __init__(self, model_dir, num_labels=3, learning_rate=5e-5):

        self.model = TFDistilBertForSequenceClassification.from_pretrained(model_dir, num_labels=num_labels)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        self.early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

    def train(self, train_dataset, val_dataset, epochs=10):
        """
        Trains the model using the training and validation datasets.
        """
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=[self.early_stopping]
        )
        return history

    def evaluate(self, test_dataset):
        """
        Evaluates the model on the testing dataset.
        """
        eval_loss, eval_acc = self.model.evaluate(test_dataset)
        print(f"Test Accuracy: {eval_acc}")
        return eval_acc

    def save_model(self, model_dir='artifacts/model/'):
        """
        Saves the trained model to a specified directory..
        """
        self.model.save_pretrained(model_dir)
        print(f"Model saved to '{model_dir}'.")

class Experiment:
    def __init__(self, s3_input_bucket, s3_input_key, s3_model_bucket, s3_model_key, s3_output_model_bucket, s3_output_model_key):

        self.s3_input_bucket = s3_input_bucket
        self.s3_input_key = s3_input_key
        self.s3_model_bucket = s3_model_bucket
        self.s3_model_key = s3_model_key
        self.s3_output_model_bucket = s3_output_model_bucket
        self.s3_output_model_key = s3_output_model_key

    def run(self):
        """
        model training pipeline
        """
        # Step 1: Import Data from S3
        print("Importing data from S3...")
        df = import_from_s3(self.s3_input_bucket, self.s3_input_key)
        if df is None:
            raise ValueError("Failed to import DataFrame from S3.")
        print("Data imported successfully.")

        # Step 2: Data Splitting
        data_processor = DataProcessor(df)
        train_df, val_df, test_df = data_processor.split_data()

        # Step 3: Download Tokenizer/Model from S3
        print("Downloading tokenizer/model from S3...")
        model_dir = 'artifacts/model/'
        if not model_from_s3(self.s3_model_bucket, self.s3_model_key, model_dir):
            raise ValueError("Failed to download tokenizer/model from S3.")
        print("Tokenizer/model downloaded successfully.")

        # Step 4: Initialize Tokenizer
        tokenizer = Tokenizer(tokenizer_path=model_dir)

        # Step 5: Create Datasets
        data_loader = DataLoader(train_df, val_df, test_df, tokenizer)
        train_dataset, val_dataset, test_dataset = data_loader.create_datasets()
        print("Datasets created successfully.")

        # Step 6: Initialize and Train Model
        model_trainer = ModelTrainer(model_dir=model_dir)
        print("Starting model training...")
        model_trainer.train(train_dataset, val_dataset)
        print("Model training completed.")

        # Step 7: Evaluate Model
        print("Evaluating model...")
        model_trainer.evaluate(test_dataset)

        # Step 8: Save Trained Model Locally
        trained_model_dir = 'artifacts/trained_model/'
        model_trainer.save_model(model_dir=trained_model_dir)

        # Step 9: Upload Trained Model to S3
        print("Uploading trained model to S3...")
        if model_to_s3(trained_model_dir, self.s3_output_model_bucket, self.s3_output_model_key):
            print("Trained model successfully uploaded to S3.")
        else:
            print("Failed to upload trained model to S3.")

        # Optional: Clean up local trained model directory
        if os.path.exists(trained_model_dir):
            shutil.rmtree(trained_model_dir)
            print(f"Local trained model directory '{trained_model_dir}' removed.")

# Example Usage
if __name__ == "__main__":
    # Define your S3 configurations
    S3_INPUT_BUCKET = 'your-input-s3-bucket-name'
    S3_INPUT_KEY = 'path/to/amazon_data.csv'  # Replace with your input CSV key in S3
    S3_MODEL_BUCKET = 'your-model-s3-bucket-name'
    S3_MODEL_KEY = 'path/to/tokenizer_model.zip'  # Replace with your tokenizer/model ZIP key in S3
    S3_OUTPUT_MODEL_BUCKET = 'your-output-s3-bucket-name'
    S3_OUTPUT_MODEL_KEY = 'path/to/trained_model.zip'  # Desired key for the trained model ZIP in S3

    # Initialize and run the experiment
    experiment = Experiment(
        s3_input_bucket=S3_INPUT_BUCKET,
        s3_input_key=S3_INPUT_KEY,
        s3_model_bucket=S3_MODEL_BUCKET,
        s3_model_key=S3_MODEL_KEY,
        s3_output_model_bucket=S3_OUTPUT_MODEL_BUCKET,
        s3_output_model_key=S3_OUTPUT_MODEL_KEY
    )

    experiment.run()