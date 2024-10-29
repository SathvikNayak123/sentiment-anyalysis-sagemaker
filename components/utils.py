import pandas as pd
from io import StringIO
import boto3
from botocore.exceptions import ClientError
import os


class Utils:
    def __init__(self):
        self.s3_client=boto3.client('s3')

    def get_data_s3(self, s3_bucket, s3_key):

        try:
            # Get the CSV object from S3
            response = self.s3_client.get_object(Bucket=s3_bucket, Key=s3_key)
            # Read the CSV data into a pandas DataFrame
            csv_content = response['Body'].read().decode('utf-8')
            df = pd.read_csv(StringIO(csv_content))
            return df
        except ClientError as e:
            print(f"Failed to retrieve data from S3: {e}")
            return None

    def put_data_s3(self, df, s3_bucket, s3_key):

        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        try:
            self.s3_client.put_object(Bucket=s3_bucket, Key=s3_key, Body=csv_buffer.getvalue())
            print(f"Data successfully uploaded to s3://{s3_bucket}/{s3_key}")
            return True
        except ClientError as e:
            print(f"Failed to upload to S3: {e}")
            return False
        
    def upload_directory_to_s3(local_directory, bucket_name, s3_directory):
        s3_client = boto3.client('s3')
        for root, _, files in os.walk(local_directory):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, local_directory)
                s3_path = os.path.join(s3_directory, relative_path).replace("\\", "/")  # For Windows compatibility
                s3_client.upload_file(local_path, bucket_name, s3_path)
                print(f"Uploaded {local_path} to s3://{bucket_name}/{s3_path}")