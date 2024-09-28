from components.data_tranform import AmazonReviewProcessor
from components.modelling import Experiment


if __name__=="__main__":

    print("-----Data Preprocess Stage-----")
    processor = AmazonReviewProcessor()
    processor.apply_preprocessing()
    processor.save_cleaned_data()

    print("-----Model Building Stage-----")
    experiment = Experiment('artifacts/data_cleaned.csv','artifacts/model/')
    experiment.run()