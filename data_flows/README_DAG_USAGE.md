# Flickr8K MinIO Pipeline DAG Usage Guide

## Overview

The `flickr8k_minio_pipeline` DAG is a streamlined Airflow pipeline that downloads the Flickr8K dataset from Hugging Face, processes it with configurable data size limits, and uploads to MinIO object storage for model training.

## Key Features

1. **Configurable Data Size**: Control how much data to download and process
2. **MinIO Integration**: Automatically uploads processed data to object storage
3. **Compatible with Model Training**: Replicates the data processing logic from `model/src/data/dataset.py`
4. **Comprehensive Reporting**: Generates detailed reports of data processing

## Configuration Options

### Dataset Size Control

You can control how much data the pipeline processes using three modes:

#### 1. Sample Mode (Default)
Uses a percentage of the total dataset:
```json
{
    "dataset_mode": "sample",
    "sample_ratio": 0.125  // Use 12.5% of dataset (~ 1000 samples)
}
```

#### 2. Count Mode
Uses a specific number of samples:
```json
{
    "dataset_mode": "count",
    "max_samples": 500  // Use exactly 500 samples
}
```

#### 3. Full Mode
Uses the entire dataset:
```json
{
    "dataset_mode": "full"
}
```

### Other Configuration Options

```json
{
    // Data size control
    "dataset_mode": "sample",           // "full", "sample", or "count"
    "max_samples": 1000,                // Max number of samples (for count/sample modes)
    "sample_ratio": 0.125,              // Fraction of dataset (for sample mode)
    
    // Data splits
    "train_split": 0.6,                 // 60% training data
    "val_split": 0.2,                   // 20% validation data
    "test_split": 0.2,                  // 20% test data
    
    // MinIO configuration
    "minio_endpoint": "localhost:9000",
    "minio_access_key": "minioadmin",
    "minio_secret_key": "minioadmin",
    "minio_bucket": "image-text-data",
    "minio_secure": false,
    
    // Processing options
    "image_format": "jpg",
    "image_size": 224,
    "overwrite_existing": false,
    
    // Storage
    "data_output_path": "/tmp/airflow_data/flickr8k"
}
```

## How to Run the DAG

### Method 1: Airflow Web UI

1. Open Airflow Web UI (typically at http://localhost:8080)
2. Find the `flickr8k_minio_pipeline` DAG
3. Click "Trigger DAG w/ Config"
4. Provide configuration in JSON format:

```json
{
    "dataset_mode": "sample",
    "max_samples": 500,
    "sample_ratio": 0.0625
}
```

### Method 2: Airflow CLI

```bash
# Run with default configuration
airflow dags trigger flickr8k_minio_pipeline

# Run with custom configuration
airflow dags trigger flickr8k_minio_pipeline \
  --conf '{"dataset_mode": "count", "max_samples": 1000}'
```

### Method 3: Using Airflow Variables (Persistent Configuration)

Set default values using Airflow Variables:

```bash
# Set default data size
airflow variables set flickr8k_dataset_mode "sample"
airflow variables set flickr8k_max_samples "1000"
airflow variables set flickr8k_sample_ratio "0.125"

# Set MinIO configuration
airflow variables set minio_endpoint "localhost:9000"
airflow variables set minio_bucket "image-text-data"
```

## Pipeline Steps

The DAG consists of the following tasks:

1. **create_directories**: Creates necessary output directories
2. **download_and_process_flickr8k**: Downloads and processes images from Hugging Face
3. **create_data_splits**: Creates train/validation/test splits
4. **setup_minio_bucket**: Creates MinIO bucket if needed
5. **upload_data_to_minio**: Uploads processed data to MinIO
6. **verify_minio_upload**: Verifies successful upload
7. **generate_pipeline_report**: Creates comprehensive report

## Data Output Structure

After successful execution, the following structure will be available in MinIO:

```
datasets/flickr8k/
├── images/
│   ├── image_000000.jpg
│   ├── image_000001.jpg
│   └── ...
├── captions.txt          // CSV format compatible with dataset.py
├── captions.json         // JSON format for easier processing
├── metadata.json         // Dataset metadata and configuration
├── train_data.json       // Training split
├── val_data.json         // Validation split
├── test_data.json        // Test split
└── pipeline_metadata.json // Pipeline execution metadata
```

## Example Usage Scenarios

### Scenario 1: Quick Testing (100 samples)
```json
{
    "dataset_mode": "count",
    "max_samples": 100
}
```

### Scenario 2: Small Development Dataset (1000 samples)
```json
{
    "dataset_mode": "count", 
    "max_samples": 1000
}
```

### Scenario 3: 10% of Full Dataset
```json
{
    "dataset_mode": "sample",
    "sample_ratio": 0.1
}
```

### Scenario 4: Full Production Dataset
```json
{
    "dataset_mode": "full"
}
```

## Monitoring and Logs

- **Task Logs**: Available in Airflow Web UI for each task
- **Pipeline Report**: Generated at the end with detailed statistics
- **MinIO Upload Verification**: Automatic verification of uploaded files
- **Data Size Tracking**: Reports show exactly how much data was processed

## Integration with Model Training

The processed data is compatible with your existing model training code:

1. **File Format**: Captions saved as `captions.txt` in CSV format
2. **Split Files**: JSON format matching your dataset loader expectations
3. **Image Processing**: Images resized and saved in the format expected by your model
4. **MinIO Access**: Your model training can directly access data from MinIO

## Performance Notes

- **Full Dataset**: ~8000 images, ~40K image-caption pairs, ~2GB
- **Sample Mode (12.5%)**: ~1000 images, ~5K pairs, ~250MB
- **Processing Time**: Depends on data size, typically 5-30 minutes
- **MinIO Upload**: Additional 2-10 minutes depending on data size 