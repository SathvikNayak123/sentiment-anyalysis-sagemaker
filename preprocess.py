import argparse
import os
from dotenv import load_dotenv, dotenv_values
import logging
from components.data_preprocess import DataProcessor
from components.data_transform import DataTransform

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clear_env_cache():
    env_vars = dotenv_values(".env")
    for key in env_vars:
        os.environ.pop(key, None)

# Clear the cache
clear_env_cache()

def main(args):
    load_dotenv()
    
    S3_RAW_BUCKET = args.s3_raw_bucket
    S3_RAW_KEY = args.s3_raw_key
    S3_CLEAN_BUCKET = args.s3_clean_bucket
    S3_CLEAN_KEY = args.s3_clean_key

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

if __name__ == "__main__":
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







