import os
import sagemaker
from sagemaker.tensorflow import TensorFlow
from dotenv import load_dotenv, dotenv_values
from sagemaker import get_execution_role
import logging
import boto3

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clear_env_cache():
    env_vars = dotenv_values(".env")
    for key in env_vars:
        os.environ.pop(key, None)

# Clear the cache
clear_env_cache()

def main():
    # Load environment variables
    load_dotenv()

    S3_MODEL_BUCKET = os.getenv('S3_MODEL_BUCKET')
    S3_MODEL_KEY = os.getenv('S3_MODEL_KEY')
    S3_CODE_BUCKET = os.getenv('S3_CODE_BUCKET')
    S3_CODE_KEY = os.getenv('S3_CODE_KEY')
    SM_ROLE = os.getenv('SM_ROLE')
    AWS_REGION=os.getenv('AWS_REGION')

    boto_session = boto3.Session(region_name=AWS_REGION)
    sagemaker_session = sagemaker.Session(boto_session=boto_session)
    logger.info("SageMaker session initialized.")
    
    # Defines where to store model artifacts in S3
    output_path = f's3://{S3_MODEL_BUCKET}/{S3_CODE_KEY}/output'
    code_location = f's3://{S3_MODEL_BUCKET}/{S3_CODE_KEY}'
    logger.info(f"Output path for model artifacts: {output_path}")
    logger.info(f"Code location in S3: {code_location}")

    print("Uploading training script and dependencies to S3...")
    sagemaker_session.upload_data(
        path='source_dir',  # Path to your source_dir
        bucket=S3_CODE_BUCKET,
        key_prefix=f'{S3_CODE_KEY}'
    )

    # Define the Estimator
    tf_estimator = TensorFlow(
        entry_point='train.py',
        role=SM_ROLE,
        image_uri='763104351884.dkr.ecr.eu-north-1.amazonaws.com/tensorflow-training:2.12.0-cpu-py310',
        instance_count=1,
        instance_type='ml.m5.xlarge',
        output_path=output_path,
        script_mode=True,
        source_dir='source_dir',
        dependencies=['source_dir/components'],
        base_job_name='sentiment-analysis-model',
        sagemaker_session=sagemaker_session,
        hyperparameters={
            's3_model_bucket': S3_MODEL_BUCKET,
            's3_model_key': S3_MODEL_KEY,
        }
    )
    
    # Define the inputs
    train_input = 'artifacts/train_data'
    val_input = 'artifacts/val_data'
    test_input = 'artifacts/test_data'

    logger.info("Starting the SageMaker Training Job...")
    # Launch the training job
    try:
        tf_estimator.fit({
            'train': train_input,
            'val': val_input,
            'test': test_input
        })
        logger.info("Training job completed successfully.")

    except Exception as e:
        print(f"Error submitting training job: {e}")

if __name__ == "__main__":
    main()
