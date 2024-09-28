import os
import pandas as pd
from components.data_collect import ScrapeData
from utils import upload_to_s3


url='https://www.amazon.com/s?k=redmi+earbuds&crid=3TZZRJIPFBP2E&sprefix=redmi+earbuds%2Caps%2C380&ref=nb_sb_noss_1'
path="chromedriver.exe"

obj=ScrapeData(url,path)
obj.scrapeReviews(20000)

os.makedirs("artifacts", exist_ok=True)
amazon_df = pd.DataFrame(obj.amazon_reviews, columns=["Review"])
amazon_df.to_csv("artifacts/amazon_data.csv", index=False)

S3_BUCKET = 'your-s3-bucket-name'  # Replace with your S3 bucket name
S3_KEY = 'path/in/bucket/amazon_data.csv'  # Replace with your desired S3 object key

upload_to_s3(amazon_df, S3_BUCKET, S3_KEY)