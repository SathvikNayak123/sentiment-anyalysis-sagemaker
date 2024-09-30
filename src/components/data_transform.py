import tensorflow as tf
from transformers import DistilBertTokenizer
from sklearn.model_selection import StratifiedShuffleSplit
import nltk
from src.components.utils import import_from_s3, upload_to_s3

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

class DataTransform:
    def __init__(self, s3_clean_bucket, s3_clean_key, s3_data_bucket, s3_train_key, s3_val_key, s3_test_key):
        """
        Initializes the DataPrep class with S3 configurations.
        """
        self.s3_clean_bucket = s3_clean_bucket
        self.s3_clean_key = s3_clean_key
        self.s3_data_bucket = s3_data_bucket
        self.s3_train_key = s3_train_key
        self.s3_val_key = s3_val_key
        self.s3_test_key = s3_test_key
        self.clean_df = None
        self.label_map = {'positive': 2, 'neutral': 1, 'negative': 0}
        self.tokenizer = None

    def import_data(self):
        """
        Imports the dataset from S3.
        """
        print("Importing data from S3...")
        self.clean_df = import_from_s3(self.s3_clean_bucket, self.s3_clean_key)
        if self.clean_df is None:
            raise ValueError("Failed to import DataFrame from S3.")
        print("Data imported successfully.")

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
        local_directory='artifacts/'
        train_dataset.save('artifacts/train_data')
        val_dataset.save('artifacts/val_data')
        test_dataset.save('artifacts/test_data')

        print("-----------saved Train, Val and Test datasets to artifacts--------\n")
    
