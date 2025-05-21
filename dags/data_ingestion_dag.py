from datetime import datetime, timedelta
import os
import tempfile
import requests
import zipfile
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from minio import Minio

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# MinIO client configuration
MINIO_ENDPOINT = 'minio:9000'
MINIO_ACCESS_KEY = 'minioadmin'
MINIO_SECRET_KEY = 'minioadmin'
MINIO_BUCKET = 'flickr8k'
USE_SSL = False

# Flickr8K dataset URL
FLICKR8K_URL = 'http://www.janderson.me/research/Flickr8k_Dataset.zip'
FLICKR8K_TEXT_URL = 'http://www.janderson.me/research/Flickr8k_text.zip'

def download_and_upload_dataset():
    """
    Downloads the Flickr8K dataset and uploads it to MinIO
    """
    # Initialize MinIO client
    minio_client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=USE_SSL
    )
    
    # Create bucket if it doesn't exist
    if not minio_client.bucket_exists(MINIO_BUCKET):
        minio_client.make_bucket(MINIO_BUCKET)
        print(f"Created bucket: {MINIO_BUCKET}")
    
    # Create temporary directory to store downloaded files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Download and process Flickr8K images
        images_zip_path = os.path.join(temp_dir, 'Flickr8k_Dataset.zip')
        print(f"Downloading Flickr8K dataset images from {FLICKR8K_URL}...")
        
        try:
            response = requests.get(FLICKR8K_URL, stream=True)
            response.raise_for_status()
            
            with open(images_zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Extract the zip file
            images_extract_dir = os.path.join(temp_dir, 'images')
            os.makedirs(images_extract_dir, exist_ok=True)
            
            with zipfile.ZipFile(images_zip_path, 'r') as zip_ref:
                zip_ref.extractall(images_extract_dir)
            
            # Upload images to MinIO
            images_dir = os.path.join(images_extract_dir, 'Flickr8k_Dataset', 'Flicker8k_Dataset')
            if os.path.exists(images_dir):
                for image_name in os.listdir(images_dir):
                    if image_name.endswith('.jpg'):
                        image_path = os.path.join(images_dir, image_name)
                        minio_object_name = f"images/{image_name}"
                        
                        # Upload the image
                        minio_client.fput_object(
                            MINIO_BUCKET, 
                            minio_object_name, 
                            image_path,
                            content_type='image/jpeg'
                        )
                        print(f"Uploaded: {minio_object_name}")
            else:
                print(f"Expected image directory not found: {images_dir}")
        
        except Exception as e:
            print(f"Error processing images: {str(e)}")
        
        # Download and process Flickr8K text annotations
        text_zip_path = os.path.join(temp_dir, 'Flickr8k_text.zip')
        print(f"Downloading Flickr8K text annotations from {FLICKR8K_TEXT_URL}...")
        
        try:
            response = requests.get(FLICKR8K_TEXT_URL, stream=True)
            response.raise_for_status()
            
            with open(text_zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Extract the zip file
            text_extract_dir = os.path.join(temp_dir, 'text')
            os.makedirs(text_extract_dir, exist_ok=True)
            
            with zipfile.ZipFile(text_zip_path, 'r') as zip_ref:
                zip_ref.extractall(text_extract_dir)
            
            # Upload text files to MinIO
            text_dir = os.path.join(text_extract_dir, 'Flickr8k_text')
            if os.path.exists(text_dir):
                for text_file in os.listdir(text_dir):
                    if text_file.endswith('.txt'):
                        text_path = os.path.join(text_dir, text_file)
                        minio_object_name = f"text/{text_file}"
                        
                        # Upload the text file
                        minio_client.fput_object(
                            MINIO_BUCKET, 
                            minio_object_name, 
                            text_path,
                            content_type='text/plain'
                        )
                        print(f"Uploaded: {minio_object_name}")
            else:
                print(f"Expected text directory not found: {text_dir}")
        
        except Exception as e:
            print(f"Error processing text files: {str(e)}")

# Define the DAG
dag = DAG(
    'flickr8k_data_ingestion',
    default_args=default_args,
    description='Downloads and uploads Flickr8K dataset to MinIO',
    schedule_interval=timedelta(days=7),  # Run weekly
    catchup=False
)

# Define the task
ingest_data_task = PythonOperator(
    task_id='ingest_data',
    python_callable=download_and_upload_dataset,
    dag=dag,
)

# Set task dependencies (only one task for now)
ingest_data_task 