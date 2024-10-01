import os
import logging
import sagemaker
from sagemaker.huggingface import HuggingFaceProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from dotenv import load_dotenv, dotenv_values
import boto3

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

    S3_RAW_BUCKET = os.getenv('S3_RAW_BUCKET')
    S3_RAW_KEY = os.getenv('S3_RAW_KEY')
    S3_CLEAN_BUCKET = os.getenv('S3_CLEAN_BUCKET')
    S3_CLEAN_KEY = os.getenv('S3_CLEAN_KEY')

    SM_ROLE = os.getenv('SM_ROLE')
    AWS_REGION = os.getenv('AWS_REGION')

    sm_session = sagemaker.Session(boto_session=boto3.Session(region_name=AWS_REGION))

    input_s3_uri = f's3://{S3_RAW_BUCKET}/{S3_RAW_KEY}'
    output_clean_s3_uri = f's3://{S3_CLEAN_BUCKET}/{S3_CLEAN_KEY}'

    # Define the script processor for data preprocessing
    hfp = HuggingFaceProcessor(
        role=SM_ROLE, 
        instance_count=1,
        instance_type='ml.m5.large',
        transformers_version='4.4.2',
        pytorch_version='1.6.0', 
        base_job_name='frameworkprocessor-hf',
        sagemaker_session=sm_session
    )

    processing_arguments = [
        '--s3-raw-bucket', S3_RAW_BUCKET,
        '--s3-raw-key', S3_RAW_KEY,
        '--s3-clean-bucket', S3_CLEAN_BUCKET,
        '--s3-clean-key', S3_CLEAN_KEY
    ]

    logger.info("Starting the data preprocessing job...")

    # Run the preprocessing job
    hfp.run(
        code='components/data_transform.py',
        source_dir='.',
        arguments=processing_arguments,
        inputs=[
            ProcessingInput(
                source=input_s3_uri,
                destination='/opt/ml/processing/input',
                input_name='raw_data'
            )
        ],
        outputs=[
            ProcessingOutput(
                source='/opt/ml/processing/output/cleaned_data',
                destination=output_clean_s3_uri,
                output_name='cleaned_data'
            )
        ],
        wait=False,
        logs=True
    )

    logger.info("Data preprocessing job completed successfully.")

if __name__ == "__main__":
    main()
