import os
import sagemaker
from sagemaker.tensorflow import TensorFlowModel
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Load environment variables
    load_dotenv()

    # Fetch configurations
    S3_BUCKET = os.getenv('S3_BUCKET')
    S3_MODEL_OUTPUT = os.getenv('S3_OUTPUT_MODEL_KEY')  # e.g., output/model.tar.gz
    ROLE = os.getenv('SAGEMAKER_ROLE')
    REGION = os.getenv('AWS_REGION', 'us-east-1')
    ENDPOINT_NAME = "sentiment-analysis-endpoint"  # Choose a unique name

    # Initialize SageMaker session
    sagemaker_session = sagemaker.Session()
    bucket = S3_BUCKET or sagemaker_session.default_bucket()

    # Define the model data path
    model_data = f"s3://{bucket}/{S3_MODEL_OUTPUT}"

    # Create a TensorFlowModel
    tf_model = TensorFlowModel(
        model_data=model_data,
        role=ROLE,
        framework_version='2.4.1',
        py_version='py38',
        sagemaker_session=sagemaker_session
    )

    # Deploy the model to an endpoint
    predictor = tf_model.deploy(
        initial_instance_count=1,
        instance_type='ml.m5.large',
        endpoint_name=ENDPOINT_NAME
    )

    logger.info(f"Model deployed to endpoint: {ENDPOINT_NAME}")

if __name__ == "__main__":
    main()
