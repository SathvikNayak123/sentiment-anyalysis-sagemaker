import os
from dotenv import load_dotenv, dotenv_values
import logging
from src.components.data_preprocess import DataProcessor
from src.components.data_transform import DataTransform

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clear_env_cache():
    env_vars = dotenv_values(".env")
    for key in env_vars:
        os.environ.pop(key, None)

# Clear the cache
clear_env_cache()

def main():
    load_dotenv()
    
    # Fetch configurations from environment variables
    S3_RAW_BUCKET = os.getenv('S3_RAW_BUCKET')
    S3_RAW_KEY = os.getenv('S3_RAW_KEY')
    S3_CLEAN_BUCKET = os.getenv('S3_CLEAN_BUCKET')
    S3_CLEAN_KEY = os.getenv('S3_CLEAN_KEY')

    logger.info(f"---Data Preprocess Starting---")

    processor = DataProcessor(S3_RAW_BUCKET, S3_RAW_KEY, S3_CLEAN_BUCKET, S3_CLEAN_KEY)
    processor.import_data_from_s3()
    classified_df = processor.process_reviews()
    processor.save_cleaned_data_to_s3(classified_df)

    logger.info(f"---Data Transformation Starting---")

    transform = DataTransform(S3_CLEAN_BUCKET, S3_CLEAN_KEY)
    transform.import_data()
    train_df, val_df, test_df = transform.preprocess_data()
    train_dataset, val_dataset, test_dataset = transform.create_datasets(train_df, val_df, test_df)
    transform.save_all_datasets(train_dataset, val_dataset, test_dataset)

    logger.info(f"---Data Preparation Complete---")

if __name__=="__main__":
    main()







