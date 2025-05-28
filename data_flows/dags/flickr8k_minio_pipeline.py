"""
Streamlined Airflow DAG for Flickr8K dataset with MinIO integration.

This DAG downloads the Flickr8K dataset from Hugging Face, processes it with
configurable data size limits, and uploads to MinIO object storage for model training.
"""

import os
import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
from airflow.configuration import conf

# Import utilities
from utils.minio_client import MinIOClient

# Import required libraries for dataset processing
from datasets import load_dataset
import pandas as pd
from PIL import Image
from tqdm import tqdm


# Default arguments for the DAG
default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

# DAG definition
dag = DAG(
    'flickr8k_minio_pipeline',
    default_args=default_args,
    description='Flickr8K dataset pipeline with MinIO object storage and configurable data size',
    schedule=None,  # Manual trigger only
    max_active_runs=1,
    catchup=False,
    tags=['data', 'flickr8k', 'minio', 'baseline'],
)


def get_pipeline_config(**context) -> Dict[str, Any]:
    """
    Get configuration for the pipeline run from Airflow Variables or DAG run conf.
    
    Parameters can be passed via:
    1. DAG run configuration when triggering manually
    2. Airflow Variables (as fallback)
    3. Default values
    """
    dag_run = context.get('dag_run')
    conf_params = dag_run.conf if dag_run and dag_run.conf else {}
    
    # Get configuration with fallbacks
    config = {
        # Dataset size configuration - NEW: configurable data amount
        'max_samples': conf_params.get('max_samples',
                                     Variable.get('flickr8k_max_samples', '1000')),
        'sample_ratio': conf_params.get('sample_ratio',
                                      float(Variable.get('flickr8k_sample_ratio', '0.125'))),  # Default to 1/8 of dataset
        'dataset_mode': conf_params.get('dataset_mode',
                                      Variable.get('flickr8k_dataset_mode', 'sample')),  # 'full', 'sample', or 'count'
        
        # Data splits
        'train_split': conf_params.get('train_split',
                                     float(Variable.get('flickr8k_train_split', '0.6'))),
        'val_split': conf_params.get('val_split',
                                   float(Variable.get('flickr8k_val_split', '0.2'))),
        'test_split': conf_params.get('test_split',
                                    float(Variable.get('flickr8k_test_split', '0.2'))),
        
        # Storage configuration
        'data_output_path': conf_params.get('data_output_path',
                                          Variable.get('flickr8k_data_path', '/tmp/airflow_data/flickr8k')),
        
        # MinIO configuration
        'minio_endpoint': conf_params.get('minio_endpoint',
                                        Variable.get('minio_endpoint', 'localhost:9000')),
        'minio_access_key': conf_params.get('minio_access_key',
                                          Variable.get('minio_access_key', 'minioadmin')),
        'minio_secret_key': conf_params.get('minio_secret_key',
                                          Variable.get('minio_secret_key', 'minioadmin')),
        'minio_bucket': conf_params.get('minio_bucket',
                                      Variable.get('minio_bucket', 'image-text-data')),
        'minio_secure': conf_params.get('minio_secure',
                                      Variable.get('minio_secure', 'false').lower() == 'true'),
        
        # Processing options
        'image_format': conf_params.get('image_format',
                                      Variable.get('flickr8k_image_format', 'jpg')),
        'image_size': conf_params.get('image_size',
                                    int(Variable.get('flickr8k_image_size', '224'))),
        'overwrite_existing': conf_params.get('overwrite_existing',
                                            Variable.get('flickr8k_overwrite', 'false').lower() == 'true'),
    }
    
    # Convert string numbers to appropriate types
    if config['max_samples'] and config['max_samples'] != 'None':
        config['max_samples'] = int(config['max_samples'])
    else:
        config['max_samples'] = None
    
    # Validate splits sum to 1.0
    total_split = config['train_split'] + config['val_split'] + config['test_split']
    if abs(total_split - 1.0) > 0.001:
        raise ValueError(f"Data splits must sum to 1.0, got {total_split}")
    
    return config


def create_directories(**context) -> str:
    """Create output directories for the dataset."""
    config = get_pipeline_config(**context)
    output_path = Path(config['data_output_path'])
    
    # Create main directories
    directories = [
        output_path,
        output_path / "processed",
        output_path / "processed" / "images",
        output_path / "logs"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Created output directories at: {output_path}")
    return str(output_path)


def download_and_process_flickr8k(**context) -> Dict[str, Any]:
    """
    Download Flickr8K dataset from Hugging Face and process it.
    This replicates the logic from model/src/data/dataset.py
    """
    config = get_pipeline_config(**context)
    output_path = Path(config['data_output_path'])
    
    print(f"🚀 Starting Flickr8K dataset download and processing...")
    print(f"📊 Configuration: {json.dumps(config, indent=2, default=str)}")
    
    # Check if data already exists and overwrite is False
    processed_images_dir = output_path / "processed" / "images"
    if processed_images_dir.exists() and any(processed_images_dir.iterdir()) and not config['overwrite_existing']:
        print("📁 Dataset already exists and overwrite_existing=False. Skipping download.")
        return {"status": "skipped", "reason": "data_already_exists"}
    
    try:
        # Load dataset from Hugging Face (same as in dataset.py)
        print("📥 Loading dataset from Hugging Face...")
        dataset = load_dataset("jxie/flickr8k", split="train")
        
        total_samples = len(dataset)
        print(f"📈 Total samples in dataset: {total_samples}")
        
        # Apply sampling based on configuration
        if config['dataset_mode'] == 'full':
            num_samples = total_samples
            print(f"🎯 Using full dataset: {num_samples} samples")
        elif config['dataset_mode'] == 'count' and config['max_samples']:
            num_samples = min(config['max_samples'], total_samples)
            print(f"🎯 Using max samples limit: {num_samples} out of {total_samples} samples")
        else:  # 'sample' mode or fallback
            if config['max_samples']:
                num_samples = min(config['max_samples'], total_samples)
            else:
                num_samples = int(total_samples * config['sample_ratio'])
            print(f"🎯 Sampling {num_samples} out of {total_samples} samples (ratio: {config['sample_ratio']})")
        
        # Select subset of data
        if num_samples < total_samples:
            dataset = dataset.select(range(num_samples))
        
        # Process and save images and captions (replicating dataset.py logic)
        captions_data = []
        images_dir = output_path / "processed" / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        print("🖼️ Processing images and captions...")
        for idx, sample in enumerate(tqdm(dataset, desc="Processing samples")):
            try:
                # Get image
                image = sample['image']
                
                # Generate filename (similar to dataset.py)
                image_filename = f"image_{idx:06d}.{config['image_format']}"
                image_path = images_dir / image_filename
                
                # Resize and save image
                if isinstance(image, Image.Image):
                    # Resize image if needed
                    target_size = (config['image_size'], config['image_size'])
                    if image.size != target_size:
                        image = image.resize(target_size, Image.Resampling.LANCZOS)
                    
                    # Convert to RGB if necessary
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    # Save image
                    image.save(image_path, quality=95, optimize=True)
                
                # Collect all captions (caption_0 to caption_4) - matching dataset.py logic
                captions = []
                for j in range(5):
                    cap_key = f'caption_{j}'
                    if cap_key in sample and sample[cap_key] and sample[cap_key].strip():
                        captions.append(sample[cap_key].strip())
                
                # Store caption data for each caption (matching dataset.py)
                for caption in captions:
                    captions_data.append({
                        'image_file': image_filename,
                        'caption': caption,
                        'image_id': f"image_{idx:06d}",
                        'sample_idx': idx
                    })
                
            except Exception as e:
                print(f"❌ Error processing sample {idx}: {str(e)}")
                continue
        
        # Save captions data as CSV (compatible with dataset.py format)
        df = pd.DataFrame([{'image': item['image_file'], 'caption': item['caption']} 
                          for item in captions_data])
        captions_file = output_path / "processed" / "captions.txt"
        df.to_csv(captions_file, index=False)
        
        # Validate captions file was created correctly
        if captions_file.exists() and captions_file.stat().st_size > 0:
            print(f"✅ Captions file created successfully: {captions_file} ({captions_file.stat().st_size} bytes)")
            print(f"📊 Sample captions data: {df.head()}")
        else:
            print(f"❌ Warning: Captions file is empty or not created: {captions_file}")
        
        # Also save as JSON for easier processing
        captions_json_file = output_path / "processed" / "captions.json"
        with open(captions_json_file, 'w') as f:
            json.dump(captions_data, f, indent=2)
        
        print(f"✅ Dataset download and processing complete! Processed {len(captions_data)} samples")
        print(f"📋 Total images: {len(set(item['image_file'] for item in captions_data))}")
        print(f"📝 Total image-caption pairs: {len(captions_data)}")
        
        # Validate we have data
        if len(captions_data) == 0:
            raise ValueError("No captions were processed! Check dataset format and sample selection.")
        
        # Save metadata
        metadata = {
            'download_timestamp': datetime.now().isoformat(),
            'dataset_source': 'huggingface:jxie/flickr8k',
            'total_samples': len(captions_data),
            'unique_images': len(set(item['image_file'] for item in captions_data)),
            'config': config,
            'processed_samples': len(captions_data),
            'image_format': config['image_format'],
            'image_size': config['image_size'],
            'dataset_mode': config['dataset_mode'],
            'sample_ratio_used': config['sample_ratio'] if config['dataset_mode'] == 'sample' else None,
            'max_samples_used': config['max_samples'] if config['dataset_mode'] in ['count', 'sample'] else None
        }
        
        metadata_file = output_path / "processed" / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return {
            "status": "success",
            "samples_processed": len(captions_data),
            "output_path": str(output_path),
            "metadata": metadata
        }
        
    except Exception as e:
        print(f"❌ Error downloading/processing dataset: {str(e)}")
        raise


def create_data_splits(**context) -> Dict[str, Any]:
    """Create train/validation/test splits from the downloaded data (replicating dataset.py logic)."""
    config = get_pipeline_config(**context)
    output_path = Path(config['data_output_path'])
    
    print("📊 Creating data splits...")
    
    # Load captions data
    captions_file = output_path / "processed" / "captions.txt"
    if not captions_file.exists():
        raise FileNotFoundError(f"Captions file not found: {captions_file}")
    
    # Check file size before attempting to read
    file_size = captions_file.stat().st_size
    print(f"📄 Captions file size: {file_size} bytes")
    
    if file_size == 0:
        raise ValueError(f"Captions file is empty: {captions_file}")
    
    try:
        # Read captions (same format as dataset.py)
        print(f"📖 Reading captions from: {captions_file}")
        df = pd.read_csv(captions_file)
        
        print(f"📊 Loaded DataFrame with shape: {df.shape}")
        print(f"📊 DataFrame columns: {list(df.columns)}")
        print(f"📊 DataFrame head:\n{df.head()}")
        
        if df.empty:
            raise ValueError("DataFrame is empty after reading captions file")
        
        if 'image' not in df.columns or 'caption' not in df.columns:
            raise ValueError(f"Expected columns 'image' and 'caption' not found. Available columns: {list(df.columns)}")
            
    except pd.errors.EmptyDataError as e:
        print(f"❌ EmptyDataError when reading captions file: {e}")
        print(f"📄 File exists: {captions_file.exists()}")
        print(f"📄 File size: {file_size} bytes")
        
        # Try to read the first few lines of the file for debugging
        try:
            with open(captions_file, 'r') as f:
                first_lines = [f.readline() for _ in range(5)]
            print(f"📄 First 5 lines of file:\n{first_lines}")
        except Exception as read_error:
            print(f"❌ Cannot read file content: {read_error}")
        
        raise
    except Exception as e:
        print(f"❌ Error reading captions file: {e}")
        raise
    
    # Group by image (same logic as dataset.py)
    grouped = df.groupby('image')['caption'].apply(list).reset_index()
    
    print(f"📝 Loaded {len(grouped)} unique images with captions")
    
    # Create splits
    import random
    random.seed(42)  # For reproducible splits
    
    # Shuffle data
    shuffled_grouped = grouped.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Calculate split indices
    n_images = len(shuffled_grouped)
    n_train = int(n_images * config['train_split'])
    n_val = int(n_images * config['val_split'])
    
    train_images = shuffled_grouped[:n_train]
    val_images = shuffled_grouped[n_train:n_train + n_val]
    test_images = shuffled_grouped[n_train + n_val:]
    
    # Create split files (compatible with dataset.py format)
    splits_info = {}
    for split_name, split_data in [('train', train_images), ('val', val_images), ('test', test_images)]:
        split_items = []
        for _, row in split_data.iterrows():
            for caption in row['caption']:
                split_items.append({
                    'image_file': row['image'],
                    'caption': caption
                })
        
        split_file = output_path / "processed" / f"{split_name}_data.json"
        with open(split_file, 'w') as f:
            json.dump(split_items, f, indent=2)
        
        splits_info[split_name] = len(split_items)
        print(f"📋 Created {split_name} split with {len(split_items)} image-caption pairs from {len(split_data)} images")
    
    # Update metadata
    metadata_file = output_path / "processed" / "metadata.json"
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    metadata['splits'] = splits_info
    metadata['splits_created'] = datetime.now().isoformat()
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return {
        "status": "success",
        "splits": splits_info,
        "total_samples": sum(splits_info.values()),
        "unique_images": {
            "train": len(train_images),
            "val": len(val_images), 
            "test": len(test_images)
        }
    }


def setup_minio_bucket(**context) -> Dict[str, Any]:
    """Create MinIO bucket if it doesn't exist."""
    config = get_pipeline_config(**context)
    
    print(f"🪣 Setting up MinIO bucket: {config['minio_bucket']}")
    
    try:
        minio_client = MinIOClient(
            endpoint=config['minio_endpoint'],
            access_key=config['minio_access_key'],
            secret_key=config['minio_secret_key'],
            secure=config['minio_secure']
        )
        
        # Create bucket if it doesn't exist
        bucket_created = minio_client.create_bucket(config['minio_bucket'])
        
        if bucket_created:
            print(f"✅ Created bucket: {config['minio_bucket']}")
        else:
            print(f"✅ Bucket already exists: {config['minio_bucket']}")
        
        # Test connectivity
        buckets = minio_client.list_buckets()
        print(f"✅ MinIO connection successful. Available buckets: {buckets}")
        
        return {
            "status": "success",
            "bucket_name": config['minio_bucket'],
            "buckets": buckets,
            "bucket_created": bucket_created
        }
        
    except Exception as e:
        print(f"❌ Failed to setup MinIO bucket: {str(e)}")
        raise


def upload_data_to_minio(**context) -> Dict[str, Any]:
    """Upload processed dataset to MinIO."""
    config = get_pipeline_config(**context)
    output_path = Path(config['data_output_path'])
    
    print(f"📤 Uploading dataset to MinIO bucket: {config['minio_bucket']}")
    
    try:
        minio_client = MinIOClient(
            endpoint=config['minio_endpoint'],
            access_key=config['minio_access_key'],
            secret_key=config['minio_secret_key'],
            secure=config['minio_secure']
        )
        
        # Upload processed data
        processed_dir = output_path / "processed"
        
        upload_results = minio_client.upload_directory(
            bucket_name=config['minio_bucket'],
            local_directory=str(processed_dir),
            object_prefix="datasets/flickr8k/"
        )
        
        print(f"✅ Upload completed: {upload_results['files_uploaded']} files")
        
        # Upload pipeline metadata with timestamp
        timestamp = datetime.now().isoformat()
        pipeline_metadata = {
            "upload_timestamp": timestamp,
            "dag_run_id": context.get('dag_run').run_id,
            "config": config,
            "upload_results": upload_results,
            "data_size_info": {
                "dataset_mode": config['dataset_mode'],
                "samples_processed": upload_results.get('samples_processed', 0),
                "max_samples": config['max_samples'],
                "sample_ratio": config['sample_ratio']
            }
        }
        
        minio_client.upload_json(
            bucket_name=config['minio_bucket'],
            object_name="datasets/flickr8k/pipeline_metadata.json",
            data=pipeline_metadata
        )
        
        return {
            "status": "success",
            "bucket_name": config['minio_bucket'],
            "files_uploaded": upload_results['files_uploaded'],
            "total_size_mb": round(upload_results['total_size'] / (1024 * 1024), 2),
            "upload_timestamp": timestamp,
            "samples_processed": upload_results.get('samples_processed', 0)
        }
        
    except Exception as e:
        print(f"❌ Failed to upload to MinIO: {str(e)}")
        raise


def verify_minio_upload(**context) -> Dict[str, Any]:
    """Verify that all files were uploaded correctly to MinIO."""
    config = get_pipeline_config(**context)
    
    print(f"🔍 Verifying MinIO upload in bucket: {config['minio_bucket']}")
    
    try:
        minio_client = MinIOClient(
            endpoint=config['minio_endpoint'],
            access_key=config['minio_access_key'],
            secret_key=config['minio_secret_key'],
            secure=config['minio_secure']
        )
        
        # List objects in the dataset directory
        objects = minio_client.list_objects(
            bucket_name=config['minio_bucket'],
            prefix="datasets/flickr8k/"
        )
        
        # Categorize files
        file_types = {
            'images': 0,
            'metadata': 0,
            'splits': 0,
            'captions': 0,
            'other': 0
        }
        
        total_size = 0
        
        for obj in objects:
            total_size += obj.size
            
            if obj.object_name.endswith(('.jpg', '.jpeg', '.png')):
                file_types['images'] += 1
            elif 'split' in obj.object_name or any(split in obj.object_name for split in ['train', 'val', 'test']):
                file_types['splits'] += 1
            elif 'caption' in obj.object_name:
                file_types['captions'] += 1
            elif obj.object_name.endswith('.json'):
                file_types['metadata'] += 1
            else:
                file_types['other'] += 1
        
        verification_results = {
            "status": "success",
            "total_objects": len(objects),
            "file_types": file_types,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "bucket_name": config['minio_bucket']
        }
        
        print(f"✅ Verification completed:")
        print(f"  📊 Total objects: {verification_results['total_objects']}")
        print(f"  🖼️ Images: {file_types['images']}")
        print(f"  📋 Split files: {file_types['splits']}")
        print(f"  📝 Caption files: {file_types['captions']}")
        print(f"  📄 Metadata files: {file_types['metadata']}")
        print(f"  📦 Total size: {verification_results['total_size_mb']} MB")
        
        return verification_results
        
    except Exception as e:
        print(f"❌ Failed to verify MinIO upload: {str(e)}")
        raise


def generate_pipeline_report(**context) -> Dict[str, Any]:
    """Generate final pipeline report with data size information."""
    config = get_pipeline_config(**context)
    
    print("📋 Generating final pipeline report...")
    
    # Get task instance results
    ti = context['ti']
    
    # Collect results from previous tasks
    download_results = ti.xcom_pull(task_ids='download_and_process_flickr8k')
    splits_results = ti.xcom_pull(task_ids='create_data_splits')
    upload_results = ti.xcom_pull(task_ids='upload_data_to_minio')
    verification_results = ti.xcom_pull(task_ids='verify_minio_upload')
    
    # Create comprehensive report
    report = {
        'pipeline_id': context.get('dag_run').run_id,
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'data_processing_summary': {
            'dataset_mode': config['dataset_mode'],
            'max_samples_requested': config['max_samples'],
            'sample_ratio_requested': config['sample_ratio'],
            'samples_actually_processed': download_results.get('samples_processed', 0) if download_results else 0,
            'splits_created': splits_results.get('splits', {}) if splits_results else {}
        },
        'results': {
            'data_download': download_results,
            'data_splits': splits_results,
            'minio_upload': upload_results,
            'upload_verification': verification_results
        },
        'status': 'success' if all([
            download_results and download_results.get('status') == 'success',
            splits_results and splits_results.get('status') == 'success',
            upload_results and upload_results.get('status') == 'success',
            verification_results and verification_results.get('status') == 'success'
        ]) else 'partial_success'
    }
    
    # Save report locally
    output_path = Path(config['data_output_path'])
    report_path = output_path / "pipeline_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Upload report to MinIO
    try:
        minio_client = MinIOClient(
            endpoint=config['minio_endpoint'],
            access_key=config['minio_access_key'],
            secret_key=config['minio_secret_key'],
            secure=config['minio_secure']
        )
        
        minio_client.upload_json(
            bucket_name=config['minio_bucket'],
            object_name=f"reports/pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            data=report
        )
        
        print("✅ Report uploaded to MinIO")
        
    except Exception as e:
        print(f"⚠️ Failed to upload report to MinIO: {str(e)}")
    
    print(f"✅ Pipeline completed with status: {report['status']}")
    print(f"📊 Data Summary:")
    print(f"  - Mode: {config['dataset_mode']}")
    print(f"  - Samples processed: {report['data_processing_summary']['samples_actually_processed']}")
    print(f"  - Files uploaded: {upload_results.get('files_uploaded', 0) if upload_results else 0}")
    
    return report


# Define tasks
create_dirs_task = PythonOperator(
    task_id='create_directories',
    python_callable=create_directories,
    dag=dag,
)

download_task = PythonOperator(
    task_id='download_and_process_flickr8k',
    python_callable=download_and_process_flickr8k,
    dag=dag,
)

create_splits_task = PythonOperator(
    task_id='create_data_splits',
    python_callable=create_data_splits,
    dag=dag,
)

setup_minio_task = PythonOperator(
    task_id='setup_minio_bucket',
    python_callable=setup_minio_bucket,
    dag=dag,
)

upload_minio_task = PythonOperator(
    task_id='upload_data_to_minio',
    python_callable=upload_data_to_minio,
    dag=dag,
)

verify_minio_task = PythonOperator(
    task_id='verify_minio_upload',
    python_callable=verify_minio_upload,
    dag=dag,
)

generate_report_task = PythonOperator(
    task_id='generate_pipeline_report',
    python_callable=generate_pipeline_report,
    dag=dag,
)

# Set task dependencies
create_dirs_task >> download_task >> create_splits_task
create_splits_task >> setup_minio_task >> upload_minio_task >> verify_minio_task >> generate_report_task 