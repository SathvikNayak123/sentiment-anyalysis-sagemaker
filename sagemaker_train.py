import os
import sagemaker
from sagemaker.tensorflow import TensorFlow
from dotenv import load_dotenv
from sagemaker import get_execution_role
import logging
import boto3

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Load environment variables
    load_dotenv()
    
    # Fetch configurations from environment variables
    S3_RAW_DATA_BUCKET = os.getenv('S3_RAW_DATA_BUCKET')
    S3_RAW_DATA_KEY = os.getenv('S3_RAW_DATA_KEY')
    S3_MODEL_BUCKET = os.getenv('S3_MODEL_BUCKET')
    S3_MODEL_KEY = os.getenv('S3_MODEL_KEY')
    S3_CLEAN_DATA_BUCKET = os.getenv('S3_CLEAN_DATA_BUCKET')
    S3_CLEAN_DATA_KEY = os.getenv('S3_CLEAN_DATA_KEY')
    SAGEMAKER_ROLE = os.getenv('AWS_ROLE')
    AWS_REGION = os.getenv('AWS_REGION')

    sagemaker_session = sagemaker.Session()
    logger.info("SageMaker session initialized.")
    
    # Defines where to store model artifacts in S3
    output_path = f's3://{S3_MODEL_BUCKET}/{S3_MODEL_KEY}/output'
    code_location = f's3://{S3_MODEL_BUCKET}/{S3_MODEL_KEY}/code'
    logger.info(f"Output path for model artifacts: {output_path}")
    logger.info(f"Code location in S3: {code_location}")

    print("Uploading training script and dependencies to S3...")
    sagemaker_session.upload_data(
        path='src',  # Path to your source_dir
        bucket=S3_MODEL_BUCKET,
        key_prefix=f'{S3_MODEL_KEY}/code'
    )

    # Define the Estimator
    tf_estimator = TensorFlow(
        entry_point='train.py',
        role=SAGEMAKER_ROLE,
        framework_version='2.16.2',
        py_version='py310',
        instance_count=1,
        instance_type='ml.m5.xlarge',
        output_path=output_path,
        script_mode=True,
        source_dir='src',
        dependencies=['components'],
        base_job_name='sentiment-analysis-model',
        sagemaker_session=sagemaker_session
    )
    
    # Define the inputs
    train_input = f's3://{S3_RAW_DATA_BUCKET}/{S3_RAW_DATA_KEY}'
    clean_input = f's3://{S3_CLEAN_DATA_BUCKET}/{S3_CLEAN_DATA_KEY}'

    logger.info("Starting the SageMaker Training Job...")
    # Launch the training job
    try:
        tf_estimator.fit({
            'raw': train_input,
            'clean': clean_input
        })
        logger.info("Training job completed successfully.")

    except Exception as e:
        print(f"Error submitting training job: {e}")

if __name__ == "__main__":
    main()
