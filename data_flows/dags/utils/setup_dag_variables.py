#!/usr/bin/env python3
"""
Setup script for Flickr8K MinIO Pipeline DAG variables.

This script sets up default Airflow variables for the flickr8k_minio_pipeline DAG.
Run this after setting up Airflow to configure sensible defaults.
"""

import subprocess
import sys
from typing import Dict, Any


def set_airflow_variable(key: str, value: str, description: str = "") -> bool:
    """Set an Airflow variable using the CLI."""
    try:
        cmd = ["airflow", "variables", "set", key, value]
        if description:
            cmd.extend(["--description", description])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Set {key} = {value}")
            return True
        else:
            print(f"❌ Failed to set {key}: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("❌ Airflow CLI not found. Make sure Airflow is installed and in PATH.")
        return False
    except Exception as e:
        print(f"❌ Error setting {key}: {str(e)}")
        return False


def setup_default_variables() -> None:
    """Setup default variables for the Flickr8K MinIO pipeline."""
    
    print("🚀 Setting up Airflow variables for Flickr8K MinIO Pipeline...")
    print()
    
    # Default variables for the DAG
    variables = {
        # Dataset configuration
        "flickr8k_dataset_mode": "sample",
        "flickr8k_max_samples": "1000", 
        "flickr8k_sample_ratio": "0.125",
        
        # Data splits
        "flickr8k_train_split": "0.6",
        "flickr8k_val_split": "0.2", 
        "flickr8k_test_split": "0.2",
        
        # Storage paths
        "flickr8k_data_path": "/tmp/airflow_data/flickr8k",
        
        # MinIO configuration (default local setup)
        "minio_endpoint": "localhost:9000",
        "minio_access_key": "minioadmin",
        "minio_secret_key": "minioadmin", 
        "minio_bucket": "image-text-data",
        "minio_secure": "false",
        
        # Processing options
        "flickr8k_image_format": "jpg",
        "flickr8k_image_size": "224",
        "flickr8k_overwrite": "false"
    }
    
    descriptions = {
        "flickr8k_dataset_mode": "Dataset size mode: 'full', 'sample', or 'count'",
        "flickr8k_max_samples": "Maximum number of samples to process (for count/sample modes)",
        "flickr8k_sample_ratio": "Fraction of dataset to use in sample mode (0.0-1.0)",
        "flickr8k_train_split": "Training data split ratio",
        "flickr8k_val_split": "Validation data split ratio",
        "flickr8k_test_split": "Test data split ratio", 
        "flickr8k_data_path": "Local output path for processed data",
        "minio_endpoint": "MinIO server endpoint (host:port)",
        "minio_access_key": "MinIO access key",
        "minio_secret_key": "MinIO secret key",
        "minio_bucket": "MinIO bucket name for storing data",
        "minio_secure": "Use HTTPS for MinIO connection (true/false)",
        "flickr8k_image_format": "Image format for saved images (jpg/png)",
        "flickr8k_image_size": "Image resize dimension (square)",
        "flickr8k_overwrite": "Overwrite existing data (true/false)"
    }
    
    successful = 0
    total = len(variables)
    
    for key, value in variables.items():
        description = descriptions.get(key, "")
        if set_airflow_variable(key, value, description):
            successful += 1
    
    print()
    print(f"📊 Setup complete: {successful}/{total} variables set successfully")
    
    if successful == total:
        print("✅ All variables set successfully!")
        print()
        print("🎯 Next steps:")
        print("1. Start MinIO: docker run -p 9000:9000 -p 9001:9001 minio/minio server /data --console-address :9001")
        print("2. Trigger the DAG: airflow dags trigger flickr8k_minio_pipeline")
        print("3. Monitor progress in Airflow Web UI")
    else:
        print("⚠️  Some variables failed to set. Check Airflow configuration.")


def setup_custom_variables(dataset_mode: str = "sample", max_samples: int = 1000, 
                          sample_ratio: float = 0.125) -> None:
    """Setup variables with custom data size configuration."""
    
    print(f"🚀 Setting up custom configuration...")
    print(f"   Dataset mode: {dataset_mode}")
    print(f"   Max samples: {max_samples}")
    print(f"   Sample ratio: {sample_ratio}")
    print()
    
    custom_vars = {
        "flickr8k_dataset_mode": dataset_mode,
        "flickr8k_max_samples": str(max_samples),
        "flickr8k_sample_ratio": str(sample_ratio)
    }
    
    for key, value in custom_vars.items():
        set_airflow_variable(key, value)
    
    print("✅ Custom configuration applied!")


def show_current_variables() -> None:
    """Show current Airflow variables related to the pipeline."""
    
    print("📋 Current Airflow variables for Flickr8K pipeline:")
    print()
    
    var_keys = [
        "flickr8k_dataset_mode", "flickr8k_max_samples", "flickr8k_sample_ratio",
        "minio_endpoint", "minio_bucket", "flickr8k_data_path"
    ]
    
    for key in var_keys:
        try:
            result = subprocess.run(
                ["airflow", "variables", "get", key], 
                capture_output=True, text=True
            )
            
            if result.returncode == 0:
                value = result.stdout.strip()
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: (not set)")
                
        except Exception as e:
            print(f"  {key}: (error reading)")


def main():
    """Main function with command line interface."""
    
    if len(sys.argv) == 1:
        # Default setup
        setup_default_variables()
        
    elif len(sys.argv) == 2:
        command = sys.argv[1]
        
        if command == "show":
            show_current_variables()
        elif command == "quick":
            setup_custom_variables("count", 100, 0.0)
        elif command == "small":
            setup_custom_variables("count", 500, 0.0)
        elif command == "medium":
            setup_custom_variables("count", 1000, 0.0)
        elif command == "full":
            setup_custom_variables("full", 8000, 1.0)
        else:
            print_usage()
            
    else:
        print_usage()


def print_usage():
    """Print usage information."""
    print("Usage: python setup_dag_variables.py [command]")
    print()
    print("Commands:")
    print("  (no args)  - Setup default variables (1000 samples)")
    print("  show       - Show current variable values") 
    print("  quick      - Quick test setup (100 samples)")
    print("  small      - Small development setup (500 samples)")
    print("  medium     - Medium baseline setup (1000 samples)")
    print("  full       - Full dataset setup")
    print()
    print("Examples:")
    print("  python setup_dag_variables.py")
    print("  python setup_dag_variables.py quick")
    print("  python setup_dag_variables.py show")


if __name__ == "__main__":
    main() 