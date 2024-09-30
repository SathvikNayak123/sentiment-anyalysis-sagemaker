import os
import logging
import sagemaker
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.processing import ScriptProcessor
from dotenv import load_dotenv, dotenv_values

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clear_env_cache():
    env_vars = dotenv_values(".env")
    for key in env_vars:
        os.environ.pop(key, None)

def main():
    # Clear the environment cache
    clear_env_cache()
    
    # Load environment variables
    load_dotenv()

    sagemaker_session=sagemaker.Session()

    # Fetch configurations from environment variables
    # Fetch configurations from environment variables
    S3_RAW_BUCKET = os.getenv('S3_RAW_BUCKET')
    S3_RAW_KEY = os.getenv('S3_RAW_KEY')
    S3_CLEAN_BUCKET = os.getenv('S3_CLEAN_BUCKET')
    S3_CLEAN_KEY = os.getenv('S3_CLEAN_KEY')
    S3_DATA_BUCKET = os.getenv('S3_DATA_BUCKET')
    S3_TRAIN_KEY = os.getenv('S3_TRAIN_KEY')
    S3_VAL_KEY = os.getenv('S3_VAL_KEY')
    S3_TEST_KEY = os.getenv('S3_TEST_KEY')

    # Define the script processor for data preprocessing
    script_processor = ScriptProcessor(
        role=os.getenv('SM_ROLE'),
        #image_uri=os.getenv('SAGEMAKER_IMAGE_URI'),
        command=['python3'],
        instance_count=1,
        instance_type='ml.m5.xlarge',
        volume_size_in_gb=1,
        max_run=3600,
        sagemaker_session=sagemaker_session,
    )

    logger.info("Starting the data preprocessing job...")

    # Run the preprocessing job
    script_processor.run(
        code='src/train.py',  # Path to your data preprocessing script
        inputs=[
            ProcessingInput(source=f's3://{S3_RAW_BUCKET}/{S3_RAW_KEY}', destination='/opt/ml/processing/input')
        ],
        outputs=[
            ProcessingOutput(source='/opt/ml/processing/output', destination=f's3://{S3_CLEAN_BUCKET}/{S3_CLEAN_KEY}'),
            ProcessingOutput(source='/opt/ml/processing/output/train', destination=f's3://{S3_DATA_BUCKET}/{S3_TRAIN_KEY}'),
            ProcessingOutput(source='/opt/ml/processing/output/val', destination=f's3://{S3_DATA_BUCKET}/{S3_VAL_KEY}'),
            ProcessingOutput(source='/opt/ml/processing/output/test', destination=f's3://{S3_DATA_BUCKET}/{S3_TEST_KEY}')
        ]
    )

    logger.info("Data preprocessing job completed successfully.")

if __name__ == "__main__":
    main()
