from dotenv import load_dotenv
from components.data_transform import DataProcessor, Split_Tokenize_Data
from components.model_train import ModelTrainer
import os

if __name__ == "__main__":
    load_dotenv()

    # Initialize parameters for DataProcessor and Split_Tokenize_Data
    bucket_name = os.getenv('bucket')
    raw_data_key = os.getenv('raw')
    clean_data_key = os.getenv('clean')
    train_data_key = os.getenv('train')
    val_data_key = os.getenv('val')
    test_data_key = os.getenv('test')
    model_key = os.getenv('model')

    # Step 1: Preprocess and classify reviews
    data_processor = DataProcessor(bucket=bucket_name, raw_key=raw_data_key, clean_key=clean_data_key)
    data_processor.import_data_from_s3()  # Load raw data
    classified_df = data_processor.ProcessReviews()  # Process and classify reviews
    data_processor.save_cleaned_data_to_s3(classified_df)  # Save processed data to S3

    # Step 2: Load cleaned data, split, tokenize, and save datasets
    split_tokenizer = Split_Tokenize_Data(
        bucket=bucket_name,
        clean_key=clean_data_key,
        train_key=train_data_key,
        val_key=val_data_key,
        test_key=test_data_key
    )

    split_tokenizer.import_data_from_s3()  # Load cleaned data
    train_df, val_df, test_df = split_tokenizer.preprocess_data()  # Split the data
    train_dataset, val_dataset, test_dataset = split_tokenizer.create_datasets(train_df, val_df, test_df)  # Tokenize and create datasets
    split_tokenizer.save_all_datasets(train_dataset, val_dataset, test_dataset)

     # Initialize the model trainer
    trainer = ModelTrainer(bucket=bucket_name, model_key=model_key)
    trainer.initialize_model()
    trainer.train(epochs=50)
    trainer.evaluate()
    trainer.save_model()
