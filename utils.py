import os
import shutil
import zipfile
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from io import StringIO
import pandas as pd


def import_from_s3(s3_bucket, s3_key):

    s3_client = boto3.client('s3')
    try:
        # Get the CSV object from S3
        response = s3_client.get_object(Bucket=s3_bucket, Key=s3_key)
        # Read the CSV data into a pandas DataFrame
        csv_content = response['Body'].read().decode('utf-8')
        df = pd.read_csv(StringIO(csv_content))
        print(f"Data successfully imported from s3://{s3_bucket}/{s3_key}")
        return df
    except NoCredentialsError:
        print("AWS credentials not available.")
        return None
    except ClientError as e:
        print(f"Failed to retrieve data from S3: {e}")
        return None

def upload_to_s3(df, s3_bucket, s3_key):

    s3_client = boto3.client('s3')
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)

    try:
        s3_client.put_object(Bucket=s3_bucket, Key=s3_key, Body=csv_buffer.getvalue())
        print(f"Data successfully uploaded to s3://{s3_bucket}/{s3_key}")
        return True
    except NoCredentialsError:
        print("AWS credentials not available.")
        return False
    except ClientError as e:
        print(f"Failed to upload to S3: {e}")
        return False

def model_to_s3(model_dir, s3_bucket, s3_key):
    """
    Compresses a local model directory into a ZIP file and uploads it to S3.
    """
    zip_file = f"{model_dir}.zip"
    
    try:
        # Create a ZIP archive of the model directory
        shutil.make_archive(model_dir, 'zip', model_dir)
        print(f"Model directory '{model_dir}' compressed into '{zip_file}'.")
        
        # Upload the ZIP file to S3
        s3_client = boto3.client('s3')
        s3_client.upload_file(zip_file, s3_bucket, s3_key)
        print(f"Model ZIP '{zip_file}' successfully uploaded to s3://{s3_bucket}/{s3_key}")
        
        # Remove the local ZIP file after upload
        os.remove(zip_file)
        print(f"Local ZIP file '{zip_file}' removed.")
        return True
    except FileNotFoundError:
        print(f"The model directory '{model_dir}' was not found.")
        return False
    except NoCredentialsError:
        print("AWS credentials not available.")
        return False
    except ClientError as e:
        print(f"Failed to upload model to S3: {e}")
        return False

def model_from_s3(s3_bucket, s3_key, local_dir):
    """
    Downloads a ZIP file from S3 and extracts it into a specified local directory.
    """
    zip_file = f"{local_dir}.zip"
    
    try:
        # Download the ZIP file from S3
        s3_client = boto3.client('s3')
        s3_client.download_file(s3_bucket, s3_key, zip_file)
        print(f"Model ZIP '{zip_file}' successfully downloaded from s3://{s3_bucket}/{s3_key}")
        
        # Extract the ZIP file into the local directory
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(local_dir)
        print(f"Model ZIP '{zip_file}' extracted to '{local_dir}'.")
        
        # Remove the downloaded ZIP file after extraction
        os.remove(zip_file)
        print(f"Local ZIP file '{zip_file}' removed.")
        return True
    except FileNotFoundError:
        print(f"The model ZIP '{zip_file}' was not found.")
        return False
    except NoCredentialsError:
        print("AWS credentials not available.")
        return False
    except ClientError as e:
        print(f"Failed to download model from S3: {e}")
        return False
    except zipfile.BadZipFile:
        print(f"The file '{zip_file}' is not a valid ZIP archive.")
        return False