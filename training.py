import os
from components.data_tranform import AmazonReviewProcessor
from components.modelling import DataPrep, ModelTrainer
from dotenv import load_dotenv

def main():
    load_dotenv()
    S3_DATA_BUCKET = os.getenv('S3_DATA_BUCKET')
    S3_DATA_KEY = os.getenv('S3_DATA_KEY')
    S3_MODEL_BUCKET = os.getenv('S3_MODEL_BUCKET')
    S3_MODEL_KEY = os.getenv('S3_MODEL_KEY')
    S3_CLEAN_DATA_BUCKET = os.getenv('S3_CLEAN_DATA_BUCKET')
    S3_CLEAN_DATA_KEY = os.getenv('S3_CLEAN_DATA_KEY')

    print("-----Data Preprocess Stage-----")
    # Initialize AmazonReviewProcessor
    processor = AmazonReviewProcessor()
    
    # Step 1: Import data from S3
    processor.import_data_from_s3(S3_DATA_BUCKET, S3_DATA_KEY)
    
    # Step 2: Process reviews (preprocess and classify)
    classified_df = processor.process_reviews()
    
    # Step 3: Save cleaned and classified data back to S3
    processor.save_cleaned_data_to_s3(classified_df, S3_CLEAN_DATA_BUCKET, S3_CLEAN_DATA_KEY)
    print("-----Data Preprocessing Completed-----\n")
    
    print("-----Model Building Stage-----")
    # Initialize DataPrep with S3 configurations for data and tokenizer/model
    data_prep = DataPrep(
        s3_input_bucket=S3_CLEAN_DATA_BUCKET,
        s3_input_key=S3_CLEAN_DATA_KEY,
        s3_model_bucket=S3_MODEL_BUCKET,
        s3_model_key=S3_MODEL_KEY
    )
    
    # Step 4: Import and preprocess data
    df = data_prep.import_data()
    train_df, val_df, test_df = data_prep.preprocess_data(df)
    
    # Step 5: Download and initialize tokenizer
    tokenizer = data_prep.download_tokenizer()
    
    # Step 6: Create TensorFlow datasets
    train_dataset, val_dataset, test_dataset = data_prep.create_datasets(train_df, val_df, test_df)
    print("-----Data Preparation for Modeling Completed-----\n")
    
    # Initialize ModelTrainer with model and S3 output configurations
    model_trainer = ModelTrainer(
        model_dir='artifacts/model/',  # Local directory where the tokenizer/model is downloaded
        s3_output_model_bucket=S3_MODEL_BUCKET,
        s3_output_model_key=S3_MODEL_KEY
    )
    
    # Step 7: Initialize the model
    model_trainer.initialize_model()
    
    # Step 8: Train the model
    model_trainer.train(train_dataset, val_dataset, epochs=10)  # Adjust epochs as needed
    
    # Step 9: Evaluate the model
    model_trainer.evaluate(test_dataset)
    
    # Step 10: Save the trained model locally
    model_trainer.save_model(trained_model_dir='artifacts/trained_model/')
    
    # Step 11: Upload the trained model to S3
    model_trainer.upload_model_to_s3(trained_model_dir='artifacts/trained_model/')
    print("-----Model Building and Uploading Completed-----")

if __name__ == "__main__":
    main()
