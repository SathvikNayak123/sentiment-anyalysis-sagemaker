import os
import traceback
import sagemaker
from sagemaker.tensorflow import TensorFlow
from dotenv import load_dotenv, dotenv_values
import logging

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

    sagemaker_session = sagemaker.Session()
    logger.info("SageMaker session initialized.")

    # Define the Estimator
    tf_estimator = TensorFlow(
        entry_point='components/model_train.py',
        role=SM_ROLE,
        framework_version='2.16.2',
        py_version='py310',
        instance_count=1,
        instance_type='ml.m5.large',
        output_path=f's3://{S3_MODEL_BUCKET}/output/',
        script_mode=True,
        source_dir='.',
        base_job_name='sentiment-analysis-model',
        sagemaker_session=sagemaker_session,
        hyperparameters={
            's3_model_bucket': S3_MODEL_BUCKET,
            's3_model_key': S3_MODEL_KEY,
        },
        wait=True,
        image_scope='training'
    )

    logger.info("Starting the SageMaker Training Job...")
    try:
        tf_estimator.fit()
        logger.info("Training job completed successfully.")

    except Exception as e:
        logger.error(traceback.format_exc())
        print(f"Error submitting training job: {e}")

if __name__ == "__main__":
    main()
