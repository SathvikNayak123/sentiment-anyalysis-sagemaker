import argparse
from src.components.model_train import ModelTrainer
import os

def main():

    S3_MODEL_BUCKET = os.getenv('S3_MODEL_BUCKET')
    S3_MODEL_KEY = os.getenv('S3_MODEL_KEY')
    
    print("-----Model Building Started-----\n")
    
    # Initialize ModelTrainer with model and S3 output configurations
    model_trainer = ModelTrainer(S3_MODEL_BUCKET,S3_MODEL_KEY)
    
    model_trainer.initialize_model()
    model_trainer.train()
    model_trainer.evaluate()
    model_trainer.save_model()

    print("-----Model Building Completed-----\n")

if __name__ == "__main__":
    main()