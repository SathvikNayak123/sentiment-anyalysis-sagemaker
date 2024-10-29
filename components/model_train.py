import tensorflow as tf
from transformers import TFDistilBertForSequenceClassification
from utils import Utils

class ModelTrainer:
    def __init__(self, bucket, model_key):
        """
        Initializes the ModelTrainer with model configurations and S3 output details.
        """
        self.train_df =  tf.data.Dataset.load('artifacts/train_data')
        self.val_df =  tf.data.Dataset.load('artifacts/val_data')
        self.test_df =  tf.data.Dataset.load('artifacts/test_data')

        self.bucket = bucket
        self.model_key = model_key

        self.model = None
        self.early_stopping = None
        self.utils  = Utils()

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

    def train(self, epochs=50):
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
        if Utils.put_model_s3('artifacts/model', self.bucket, self.model_key):
            print("Trained model successfully uploaded to S3.")
        else:
            print("Failed to upload trained model to S3.")