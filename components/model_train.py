import subprocess
import sys
import tensorflow as tf
from transformers import TFDistilBertForSequenceClassification
import argparse
import os
import shutil
import boto3
from botocore.exceptions import ClientError

class Utils:
    def __init__(self, model_dir, s3_bucket, s3_key):
        
        self.s3_client = boto3.client('s3')

        self.model_dir = model_dir
        self.s3_model_bucket = s3_bucket
        self.s3_model_key = s3_key

        self.zip_file = f"{model_dir}.zip"

    def put_model_s3(self):
        try:
            shutil.make_archive(self.model_dir, 'zip', self.model_dir)
            self.s3_client.upload_file(self.zip_file, self.s3_model_bucket, self.s3_model_key)
            print(f"Model ZIP '{self.zip_file}' successfully uploaded to s3://{self.s3_model_bucket}/{self.s3_model_key}")
            
            # Remove the local ZIP file after upload
            os.remove(self.zip_file)
            print(f"Local ZIP file '{self.zip_file}' removed.")
            return True
        
        except FileNotFoundError:
            print(f"The model directory '{self.model_dir}' was not found.")
            return False
        except ClientError as e:
            print(f"Failed to upload model to S3: {e}")
            return False
        

class ModelTrainer:
    def __init__(self, s3_model_bucket, s3_model_key):
        """
        Initializes the ModelTrainer with model configurations and S3 output details.
        """
        self.train_df =  tf.data.Dataset.load('artifacts/train_data')
        self.val_df =  tf.data.Dataset.load('artifacts/val_data')
        self.test_df =  tf.data.Dataset.load('artifacts/test_data')

        self.s3_model_bucket = s3_model_bucket
        self.s3_model_key = s3_model_key

        self.model = None
        self.early_stopping = None

    def initialize_model(self):
        """
        Loads the DistilBERT model from HuggingFace for sequence classification.
        """
        print("Initializing the model...")
        self.model = TFDistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            num_labels=3
        )
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        self.early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=2,
            restore_best_weights=True
        )
        print("Model initialized successfully.")

    def train(self, epochs=10):
        """
        Trains the model using the training and validation datasets.
        """
        print("Starting model training...")
        history = self.model.fit(
            self.train_df,
            validation_data=self.val_df,
            epochs=epochs,
            callbacks=[self.early_stopping]
        )
        print("Model training completed.")
        return history

    def evaluate(self):
        """
        Evaluates the model on the testing dataset.
        """
        print("Evaluating the model...")
        eval_loss, eval_acc = self.model.evaluate(self.test_df)
        print(f"Test Accuracy: {eval_acc}")
        return eval_loss, eval_acc

    def save_model(self):
        """
        Saves the trained model to local and S3
        """
        self.model.save_pretrained('artifacts/model')
 
        print("Uploading trained model to S3...")
        if Utils.put_model_s3('artifacts/model', self.s3_model_bucket, self.s3_model_key):
            print("Trained model successfully uploaded to S3.")
        else:
            print("Failed to upload trained model to S3.")

def main(args):

    S3_MODEL_BUCKET = args.s3_model_bucket
    S3_MODEL_KEY = args.s3_model_key
    
    print("-----Model Building Started-----\n")
    
    # Initialize ModelTrainer with model and S3 output configurations
    model_trainer = ModelTrainer(S3_MODEL_BUCKET,S3_MODEL_KEY)
    
    model_trainer.initialize_model()
    model_trainer.train()
    model_trainer.evaluate()
    model_trainer.save_model()

    print("-----Model Building Completed-----\n")

if __name__ == "__main__":
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    parser = argparse.ArgumentParser(description="SageMaker Training Script")

    # Input S3 paths
    parser.add_argument(
        "--s3-model-bucket",
        type=str,
        required=True
    )
    parser.add_argument(
        "--s3-model-key",
        type=str,
        required=True
    )
    args = parser.parse_args()
    main(args)
