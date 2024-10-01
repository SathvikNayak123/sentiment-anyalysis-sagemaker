import os
import pandas as pd
from components.data_collect import ScrapeData
from components.utils import upload_to_s3
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

url='https://www.amazon.com/s?k=redmi+earbuds&crid=3TZZRJIPFBP2E&sprefix=redmi+earbuds%2Caps%2C380&ref=nb_sb_noss_1'
path="chromedriver.exe"

obj=ScrapeData(url,path)
obj.scrapeReviews(20000)

logger.info(f"---Data Collection Complete---")

os.makedirs("artifacts", exist_ok=True)
amazon_df = pd.DataFrame(obj.amazon_reviews, columns=["Review"])
amazon_df.to_csv("artifacts/amazon_data.csv", index=False)

S3_RAW_BUCKET = os.getenv('S3_RAW_BUCKET')
S3_RAW_KEY = os.getenv('S3_RAW_KEY')

upload_to_s3(amazon_df, S3_RAW_BUCKET, S3_RAW_KEY)

logger.info(f"---Uploaded Raw Data to S3---")