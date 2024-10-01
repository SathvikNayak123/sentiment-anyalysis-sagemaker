import argparse
from components.model_train import ModelTrainer
import os

def main(args):

    S3_MODEL_BUCKET = args.s3_model_bucket
    S3_MODEL_KEY = args.s3_model_key
    
    print("-----Model Building Started-----\n")
    
    # Initialize ModelTrainer with model and S3 output configurations
    model_trainer = ModelTrainer(S3_MODEL_BUCKET,S3_MODEL_KEY)
    
    model_trainer.initialize_model()
    model_trainer.train()
    model_trainer.evaluate()
    model_trainer.save_model()

    print("-----Model Building Completed-----\n")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="SageMaker Training Script")

    # Input S3 paths
    parser.add_argument(
        "--s3-model-bucket",
        type=str,
        required=True
    )
    parser.add_argument(
        "--s3-model-key",
        type=str,
        required=True
    )
    args = parser.parse_args()
    main(args)