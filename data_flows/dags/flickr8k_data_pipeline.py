"""
Airflow DAG for Flickr8K dataset download and processing pipeline.

This DAG downloads the Flickr8K dataset from Hugging Face and processes it
for the image-text ranking model. It supports configurable parameters for
partial dataset downloads and flexible data processing.
"""

import os
import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.models import Variable
from airflow.configuration import conf

# Import dataset utilities
import sys
sys.path.append('/opt/airflow/dags/utils')

from datasets import load_dataset
import pandas as pd
from PIL import Image
import requests
from tqdm import tqdm


# Default arguments for the DAG
default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

# DAG definition
dag = DAG(
    'flickr8k_data_pipeline',
    default_args=default_args,
    description='Download and process Flickr8K dataset for image-text ranking',
    schedule=None,  # Manual trigger only
    max_active_runs=1,
    catchup=False,
    tags=['data', 'flickr8k', 'image-text-ranking'],
)


def get_dag_config(**context) -> Dict[str, Any]:
    """
    Get configuration for the DAG run from Airflow Variables or DAG run conf.
    
    Parameters can be passed via:
    1. DAG run configuration when triggering manually
    2. Airflow Variables (as fallback)
    3. Default values
    """
    dag_run = context.get('dag_run')
    conf_params = dag_run.conf if dag_run and dag_run.conf else {}
    
    # Get configuration with fallbacks
    config = {
        # Dataset size configuration
        'dataset_size': conf_params.get('dataset_size', 
                                      Variable.get('flickr8k_dataset_size', 'full')),
        'max_samples': conf_params.get('max_samples',
                                     Variable.get('flickr8k_max_samples', None)),
        'sample_ratio': conf_params.get('sample_ratio',
                                      float(Variable.get('flickr8k_sample_ratio', '1.0'))),
        
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
        'use_object_storage': conf_params.get('use_object_storage',
                                            Variable.get('use_object_storage', 'false').lower() == 'true'),
        'storage_bucket': conf_params.get('storage_bucket',
                                        Variable.get('storage_bucket', 'image-text-data')),
        
        # Processing options
        'image_format': conf_params.get('image_format',
                                      Variable.get('flickr8k_image_format', 'jpg')),
        'image_size': conf_params.get('image_size',
                                    Variable.get('flickr8k_image_size', '224')),
        'overwrite_existing': conf_params.get('overwrite_existing',
                                            Variable.get('flickr8k_overwrite', 'false').lower() == 'true'),
    }
    
    # Convert string numbers to appropriate types
    if config['max_samples'] and config['max_samples'] != 'None':
        config['max_samples'] = int(config['max_samples'])
    else:
        config['max_samples'] = None
        
    config['image_size'] = int(config['image_size'])
    
    # Validate splits sum to 1.0
    total_split = config['train_split'] + config['val_split'] + config['test_split']
    if abs(total_split - 1.0) > 0.001:
        raise ValueError(f"Data splits must sum to 1.0, got {total_split}")
    
    return config


def create_output_directories(**context) -> str:
    """Create output directories for the dataset."""
    config = get_dag_config(**context)
    output_path = Path(config['data_output_path'])
    
    # Create main directories
    directories = [
        output_path,
        output_path / "raw",
        output_path / "processed",
        output_path / "processed" / "images",
        output_path / "logs"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    print(f"Created output directories at: {output_path}")
    return str(output_path)


def download_flickr8k_dataset(**context) -> Dict[str, Any]:
    """Download Flickr8K dataset from Hugging Face."""
    config = get_dag_config(**context)
    output_path = Path(config['data_output_path'])
    
    print(f"Starting Flickr8K dataset download...")
    print(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Check if data already exists and overwrite is False
    processed_images_dir = output_path / "processed" / "images"
    if processed_images_dir.exists() and any(processed_images_dir.iterdir()) and not config['overwrite_existing']:
        print("Dataset already exists and overwrite_existing=False. Skipping download.")
        return {"status": "skipped", "reason": "data_already_exists"}
    
    try:
        # Load dataset from Hugging Face
        print("Loading dataset from Hugging Face...")
        dataset = load_dataset("jxie/flickr8k", split="train")
        
        total_samples = len(dataset)
        print(f"Total samples in dataset: {total_samples}")
        
        # Apply sampling if configured
        if config['dataset_size'] == 'sample' or config['sample_ratio'] < 1.0:
            if config['max_samples']:
                num_samples = min(config['max_samples'], total_samples)
            else:
                num_samples = int(total_samples * config['sample_ratio'])
            
            print(f"Sampling {num_samples} out of {total_samples} samples")
            dataset = dataset.select(range(num_samples))
        else:
            num_samples = total_samples
            print(f"Using full dataset: {num_samples} samples")
        
        # Process and save images and captions
        captions_data = []
        images_dir = output_path / "processed" / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        print("Processing images and captions...")
        for idx, sample in enumerate(tqdm(dataset, desc="Processing samples")):
            try:
                # Get image and caption
                image = sample['image']
                caption = sample['caption']
                
                # Generate filename
                image_filename = f"flickr8k_{idx:06d}.{config['image_format']}"
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
                
                # Store caption data
                captions_data.append({
                    'image_file': image_filename,
                    'caption': caption,
                    'image_id': f"flickr8k_{idx:06d}",
                    'sample_idx': idx
                })
                
            except Exception as e:
                print(f"Error processing sample {idx}: {str(e)}")
                continue
        
        # Save captions data
        captions_file = output_path / "processed" / "captions.json"
        with open(captions_file, 'w') as f:
            json.dump(captions_data, f, indent=2)
        
        # Save metadata
        metadata = {
            'download_timestamp': datetime.now().isoformat(),
            'dataset_source': 'huggingface:jxie/flickr8k',
            'total_samples': len(captions_data),
            'config': config,
            'processed_samples': len(captions_data),
            'image_format': config['image_format'],
            'image_size': config['image_size']
        }
        
        metadata_file = output_path / "processed" / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Dataset download complete! Processed {len(captions_data)} samples")
        return {
            "status": "success",
            "samples_processed": len(captions_data),
            "output_path": str(output_path),
            "metadata": metadata
        }
        
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        raise


def create_data_splits(**context) -> Dict[str, Any]:
    """Create train/validation/test splits from the downloaded data."""
    config = get_dag_config(**context)
    output_path = Path(config['data_output_path'])
    
    print("Creating data splits...")
    
    # Load captions data
    captions_file = output_path / "processed" / "captions.json"
    if not captions_file.exists():
        raise FileNotFoundError(f"Captions file not found: {captions_file}")
    
    with open(captions_file, 'r') as f:
        captions_data = json.load(f)
    
    print(f"Loaded {len(captions_data)} caption entries")
    
    # Create splits
    import random
    random.seed(42)  # For reproducible splits
    
    # Shuffle data
    shuffled_data = captions_data.copy()
    random.shuffle(shuffled_data)
    
    # Calculate split indices
    total_samples = len(shuffled_data)
    train_end = int(total_samples * config['train_split'])
    val_end = train_end + int(total_samples * config['val_split'])
    
    # Create splits
    splits = {
        'train': shuffled_data[:train_end],
        'val': shuffled_data[train_end:val_end],
        'test': shuffled_data[val_end:]
    }
    
    # Save split files
    for split_name, split_data in splits.items():
        split_file = output_path / "processed" / f"{split_name}_data.json"
        with open(split_file, 'w') as f:
            json.dump(split_data, f, indent=2)
        print(f"Created {split_name} split with {len(split_data)} samples")
    
    # Update metadata
    metadata_file = output_path / "processed" / "metadata.json"
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    metadata['splits'] = {split: len(data) for split, data in splits.items()}
    metadata['splits_created'] = datetime.now().isoformat()
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return {
        "status": "success",
        "splits": {split: len(data) for split, data in splits.items()},
        "total_samples": total_samples
    }


def validate_dataset(**context) -> Dict[str, Any]:
    """Validate the processed dataset."""
    config = get_dag_config(**context)
    output_path = Path(config['data_output_path'])
    
    print("Validating processed dataset...")
    
    validation_results = {
        "status": "success",
        "errors": [],
        "warnings": [],
        "stats": {}
    }
    
    try:
        # Check metadata file
        metadata_file = output_path / "processed" / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            validation_results["stats"]["metadata"] = metadata
        else:
            validation_results["errors"].append("Metadata file missing")
        
        # Check split files
        splits = ['train', 'val', 'test']
        for split in splits:
            split_file = output_path / "processed" / f"{split}_data.json"
            if split_file.exists():
                with open(split_file, 'r') as f:
                    split_data = json.load(f)
                validation_results["stats"][f"{split}_samples"] = len(split_data)
                
                # Validate first few samples
                for i, sample in enumerate(split_data[:5]):
                    image_path = output_path / "processed" / "images" / sample['image_file']
                    if not image_path.exists():
                        validation_results["errors"].append(
                            f"Missing image file: {sample['image_file']} in {split} split"
                        )
            else:
                validation_results["errors"].append(f"Missing split file: {split}")
        
        # Check images directory
        images_dir = output_path / "processed" / "images"
        if images_dir.exists():
            image_files = list(images_dir.glob(f"*.{config['image_format']}"))
            validation_results["stats"]["total_images"] = len(image_files)
            
            if len(image_files) == 0:
                validation_results["errors"].append("No image files found")
        else:
            validation_results["errors"].append("Images directory not found")
        
        # Set overall status
        if validation_results["errors"]:
            validation_results["status"] = "failed"
        elif validation_results["warnings"]:
            validation_results["status"] = "warning"
        
        print(f"Validation complete. Status: {validation_results['status']}")
        if validation_results["errors"]:
            print(f"Errors: {validation_results['errors']}")
        if validation_results["warnings"]:
            print(f"Warnings: {validation_results['warnings']}")
        
        return validation_results
        
    except Exception as e:
        validation_results["status"] = "failed"
        validation_results["errors"].append(f"Validation failed: {str(e)}")
        print(f"Validation failed: {str(e)}")
        return validation_results


def upload_to_object_storage(**context) -> Dict[str, Any]:
    """Upload processed data to object storage (optional step)."""
    config = get_dag_config(**context)
    
    if not config['use_object_storage']:
        print("Object storage upload skipped (use_object_storage=False)")
        return {"status": "skipped", "reason": "object_storage_disabled"}
    
    # This is a placeholder for object storage integration
    # In a real implementation, you would integrate with MinIO, S3, etc.
    print(f"Would upload to object storage bucket: {config['storage_bucket']}")
    print("Object storage integration not implemented in this version")
    
    return {
        "status": "not_implemented",
        "message": "Object storage integration to be implemented"
    }


# Define task dependencies
create_dirs_task = PythonOperator(
    task_id='create_output_directories',
    python_callable=create_output_directories,
    dag=dag,
)

download_task = PythonOperator(
    task_id='download_flickr8k_dataset',
    python_callable=download_flickr8k_dataset,
    dag=dag,
)

create_splits_task = PythonOperator(
    task_id='create_data_splits',
    python_callable=create_data_splits,
    dag=dag,
)

validate_task = PythonOperator(
    task_id='validate_dataset',
    python_callable=validate_dataset,
    dag=dag,
)

upload_task = PythonOperator(
    task_id='upload_to_object_storage',
    python_callable=upload_to_object_storage,
    dag=dag,
)

# Set task dependencies
create_dirs_task >> download_task >> create_splits_task >> validate_task >> upload_task 