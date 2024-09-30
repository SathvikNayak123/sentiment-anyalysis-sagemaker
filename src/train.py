import os
from components.data_tranform import AmazonReviewProcessor
from components.modelling import DataPrep, ModelTrainer
from dotenv import load_dotenv

def main():
    load_dotenv()
    S3_RAW_DATA_BUCKET = os.getenv('S3_RAW_DATA_BUCKET')
    S3_RAW_DATA_KEY = os.getenv('S3_RAW_DATA_KEY')
    S3_MODEL_BUCKET = os.getenv('S3_MODEL_BUCKET')
    S3_MODEL_KEY = os.getenv('S3_MODEL_KEY')
    S3_CLEAN_DATA_BUCKET = os.getenv('S3_CLEAN_DATA_BUCKET')
    S3_CLEAN_DATA_KEY = os.getenv('S3_CLEAN_DATA_KEY')

    print("-----Data Preprocess Stage-----\n")

    processor = AmazonReviewProcessor()
    # Import data from S3
    processor.import_data_from_s3(S3_RAW_DATA_BUCKET, S3_RAW_DATA_KEY)
    # Process and pseudo-label reviews
    classified_df = processor.process_reviews()
    # Save cleaned and classified data back to S3
    processor.save_cleaned_data_to_s3(classified_df, S3_CLEAN_DATA_BUCKET, S3_CLEAN_DATA_KEY)

    print("-----Data Preprocessing Completed-----\n")
    
    print("-----Data Preparation for Modeling-----\n")

    # Initialize DataPrep with S3 configurations for data and tokenizer/model
    data_prep = DataPrep(
        s3_input_bucket=S3_CLEAN_DATA_BUCKET,
        s3_input_key=S3_CLEAN_DATA_KEY,
        s3_model_bucket=S3_MODEL_BUCKET,
        s3_model_key=S3_MODEL_KEY
    )

    # Import and preprocess data
    df = data_prep.import_data()
    # Split dataset
    train_df, val_df, test_df = data_prep.preprocess_data(df)
    # Create TensorFlow datasets
    train_dataset, val_dataset, test_dataset = data_prep.create_datasets(train_df, val_df, test_df)

    print("-----Data Preparation for Modeling Completed-----\n")
    
    # Initialize ModelTrainer with model and S3 output configurations
    model_trainer = ModelTrainer(  # Local directory where the tokenizer/model is downloaded
        s3_output_model_bucket=S3_MODEL_BUCKET,
        s3_output_model_key=S3_MODEL_KEY
    )
    
    # Initialize the model
    model_trainer.initialize_model()
    # Train the model
    model_trainer.train(train_dataset, val_dataset, epochs=10)
    # Evaluate the model
    model_trainer.evaluate(test_dataset)
    # Save the trained model locally
    model_trainer.save_model()
    # Upload the trained model to S3
    model_trainer.upload_model_to_s3()

    print("-----Model Building Completed-----\n")

if __name__ == "__main__":
    main()