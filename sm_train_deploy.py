import os
import sys
import traceback
import sagemaker
from sagemaker.tensorflow import TensorFlow
from dotenv import load_dotenv, dotenv_values
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
    SM_ROLE = os.getenv('SM_ROLE')
    AWS_REGION = os.getenv('AWS_REGION')
    ENDPOINT_NAME = os.getenv('ENDPOINT_NAME')

    sm_session = sagemaker.Session(boto_session=boto3.Session(region_name=AWS_REGION))
    logger.info("SageMaker session initialized.")

    model_path = f"s3://{S3_MODEL_BUCKET}/{S3_MODEL_KEY}"

    # Define the Estimator
    tf_estimator = TensorFlow(
        entry_point='components/model_train.py',
        role=SM_ROLE,
        framework_version='2.12',
        py_version='py310',
        instance_count=1,
        instance_type='ml.p2.xlarge',
        output_path=model_path,
        script_mode=True,
        source_dir='.',
        base_job_name='sentiment-analysis-model',
        sagemaker_session=sm_session,
        hyperparameters={
            's3_model_bucket': S3_MODEL_BUCKET,
            's3_model_key': S3_MODEL_KEY,
        },
        wait=True
    )

    logger.info("Starting the SageMaker Training Job...")
    try:
        tf_estimator.fit()
        logger.info("Training job completed successfully.")
        
        # Proceed to deployment only if training succeeds
        tf_estimator.deploy(
            initial_instance_count=1,
            instance_type='ml.p2.xlarge',
            endpoint_name=ENDPOINT_NAME
        )
        logger.info(f"Model deployed to endpoint: {ENDPOINT_NAME}")
        
    except Exception as e:
        logger.error("An error occurred during the training or deployment process.")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
