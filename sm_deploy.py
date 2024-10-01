import os
import sagemaker
from sagemaker.tensorflow import TensorFlowModel
from dotenv import load_dotenv
import logging
import boto3

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Load environment variables
    load_dotenv()

    # Fetch configurations
    S3_MODEL_BUCKET = os.getenv('S3_MODEL_BUCKET')
    S3_MODEL_KEY = os.getenv('S3_MODEL_KEY')
    ROLE = os.getenv('SM_ROLE')
    ENDPOINT_NAME = "sentiment-analysis-endpoint"
    AWS_REGION = os.getenv('AWS_REGION')

    # Initialize SageMaker session
    sagemaker_session = sagemaker.Session(boto_session=boto3.Session(region_name=AWS_REGION))

    # Define the model data path
    model_data = f"s3://{S3_MODEL_BUCKET}/{S3_MODEL_KEY}"

    # Create a TensorFlowModel
    tf_model = TensorFlowModel(
        model_data=model_data,
        role=ROLE,
        framework_version='2.16.2',
        py_version='py310',
        sagemaker_session=sagemaker_session
    )

    # Deploy the model to an endpoint
    tf_model.deploy(
        initial_instance_count=1,
        instance_type='ml.m5.large',
        endpoint_name=ENDPOINT_NAME
    )

    logger.info(f"Model deployed to endpoint: {ENDPOINT_NAME}")

if __name__ == "__main__":
    main()
