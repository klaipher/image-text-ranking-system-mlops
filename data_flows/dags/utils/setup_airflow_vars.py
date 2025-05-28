#!/usr/bin/env python3
"""
Utility script to set up Airflow variables for the Flickr8K data pipeline.

This script helps configure Airflow variables needed for the data pipeline
either via command line or programmatically.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any


DEFAULT_VARIABLES = {
    # Basic Configuration
    'flickr8k_dataset_size': 'full',
    'flickr8k_sample_ratio': '1.0',
    'flickr8k_max_samples': None,
    
    # Data Splits
    'flickr8k_train_split': '0.6',
    'flickr8k_val_split': '0.2',
    'flickr8k_test_split': '0.2',
    
    # Storage Configuration
    'flickr8k_data_path': '/tmp/airflow_data/flickr8k',
    'model_data_path': '../model/data',
    'use_object_storage': 'true',
    'storage_bucket': 'image-text-data',
    
    # MinIO Configuration
    'minio_endpoint': 'minio:9000',
    'minio_access_key': 'minioadmin',
    'minio_secret_key': 'minioadmin',
    'minio_bucket': 'image-text-data',
    'minio_secure': 'false',
    
    # Processing Options
    'flickr8k_image_format': 'jpg',
    'flickr8k_image_size': '224',
    'flickr8k_overwrite': 'false',
}

SAMPLE_CONFIGURATIONS = {
    'full_dataset': {
        'flickr8k_dataset_size': 'full',
        'flickr8k_sample_ratio': '1.0',
        'flickr8k_max_samples': None,
    },
    'small_sample': {
        'flickr8k_dataset_size': 'sample',
        'flickr8k_sample_ratio': '0.1',
        'flickr8k_max_samples': '800',
    },
    'medium_sample': {
        'flickr8k_dataset_size': 'sample',
        'flickr8k_sample_ratio': '0.25',
        'flickr8k_max_samples': '2000',
    },
    'development': {
        'flickr8k_dataset_size': 'sample',
        'flickr8k_sample_ratio': '0.05',
        'flickr8k_max_samples': '100',
        'flickr8k_image_size': '128',  # Smaller images for faster processing
    }
}


def run_airflow_command(cmd: list) -> bool:
    """Run an Airflow CLI command."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✓ Command succeeded: {' '.join(cmd)}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Command failed: {' '.join(cmd)}")
        print(f"Error: {e.stderr}")
        return False
    except FileNotFoundError:
        print("✗ Airflow CLI not found. Make sure Airflow is installed and in PATH.")
        return False


def set_airflow_variable(key: str, value: str) -> bool:
    """Set an Airflow variable."""
    cmd = ['airflow', 'variables', 'set', key, value]
    return run_airflow_command(cmd)


def get_airflow_variable(key: str) -> str:
    """Get an Airflow variable."""
    try:
        cmd = ['airflow', 'variables', 'get', key]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None


def list_airflow_variables() -> Dict[str, str]:
    """List all Airflow variables."""
    try:
        cmd = ['airflow', 'variables', 'list', '--output', 'json']
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        variables = json.loads(result.stdout)
        return {var['key']: var['val'] for var in variables}
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        print("Failed to list variables")
        return {}


def setup_default_variables() -> None:
    """Set up default Airflow variables."""
    print("Setting up default Airflow variables for Flickr8K pipeline...")
    
    success_count = 0
    total_count = len(DEFAULT_VARIABLES)
    
    for key, value in DEFAULT_VARIABLES.items():
        if value is not None:
            if set_airflow_variable(key, str(value)):
                success_count += 1
        else:
            print(f"⚠ Skipping variable with None value: {key}")
    
    print(f"\nSetup complete: {success_count}/{total_count} variables set successfully")


def setup_configuration(config_name: str) -> None:
    """Set up a predefined configuration."""
    if config_name not in SAMPLE_CONFIGURATIONS:
        print(f"Unknown configuration: {config_name}")
        print(f"Available configurations: {list(SAMPLE_CONFIGURATIONS.keys())}")
        return
    
    config = SAMPLE_CONFIGURATIONS[config_name]
    print(f"Setting up '{config_name}' configuration...")
    
    success_count = 0
    for key, value in config.items():
        if value is not None:
            if set_airflow_variable(key, str(value)):
                success_count += 1
        else:
            # Delete the variable if value is None
            cmd = ['airflow', 'variables', 'delete', key]
            run_airflow_command(cmd)
    
    print(f"Configuration '{config_name}' setup complete: {success_count}/{len(config)} variables set")


def show_current_config() -> None:
    """Show current configuration."""
    print("Current Airflow Variables Configuration:")
    print("=" * 50)
    
    variables = list_airflow_variables()
    
    # Group variables by category
    categories = {
        'Basic Configuration': ['flickr8k_dataset_size', 'flickr8k_sample_ratio', 'flickr8k_max_samples'],
        'Data Splits': ['flickr8k_train_split', 'flickr8k_val_split', 'flickr8k_test_split'],
        'Storage': ['flickr8k_data_path', 'model_data_path', 'use_object_storage', 'storage_bucket'],
        'Processing': ['flickr8k_image_format', 'flickr8k_image_size', 'flickr8k_overwrite']
    }
    
    for category, keys in categories.items():
        print(f"\n{category}:")
        for key in keys:
            value = variables.get(key, 'NOT SET')
            print(f"  {key}: {value}")


def export_config_to_json(filename: str) -> None:
    """Export current configuration to JSON file."""
    variables = list_airflow_variables()
    
    # Filter only our variables
    our_variables = {k: v for k, v in variables.items() 
                    if k.startswith('flickr8k_') or k in ['model_data_path', 'use_object_storage', 'storage_bucket']}
    
    with open(filename, 'w') as f:
        json.dump(our_variables, f, indent=2)
    
    print(f"Configuration exported to {filename}")


def import_config_from_json(filename: str) -> None:
    """Import configuration from JSON file."""
    try:
        with open(filename, 'r') as f:
            config = json.load(f)
        
        print(f"Importing configuration from {filename}...")
        success_count = 0
        
        for key, value in config.items():
            if set_airflow_variable(key, str(value)):
                success_count += 1
        
        print(f"Import complete: {success_count}/{len(config)} variables set")
        
    except FileNotFoundError:
        print(f"File not found: {filename}")
    except json.JSONDecodeError:
        print(f"Invalid JSON file: {filename}")


def test_dag_config() -> None:
    """Test DAG configuration by showing what would be used."""
    print("Testing DAG Configuration...")
    print("=" * 50)
    
    # Simulate what get_dag_config would return
    variables = list_airflow_variables()
    
    config = {
        'dataset_size': variables.get('flickr8k_dataset_size', 'full'),
        'sample_ratio': float(variables.get('flickr8k_sample_ratio', '1.0')),
        'max_samples': variables.get('flickr8k_max_samples', None),
        'train_split': float(variables.get('flickr8k_train_split', '0.6')),
        'val_split': float(variables.get('flickr8k_val_split', '0.2')),
        'test_split': float(variables.get('flickr8k_test_split', '0.2')),
        'data_output_path': variables.get('flickr8k_data_path', '/tmp/airflow_data/flickr8k'),
        'image_format': variables.get('flickr8k_image_format', 'jpg'),
        'image_size': int(variables.get('flickr8k_image_size', '224')),
        'overwrite_existing': variables.get('flickr8k_overwrite', 'false').lower() == 'true',
    }
    
    if config['max_samples'] and config['max_samples'] != 'None':
        config['max_samples'] = int(config['max_samples'])
    else:
        config['max_samples'] = None
    
    # Validate splits
    total_split = config['train_split'] + config['val_split'] + config['test_split']
    
    print("Effective Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print(f"\nValidation:")
    print(f"  Total split ratio: {total_split:.3f} {'✓' if abs(total_split - 1.0) < 0.001 else '✗'}")
    
    if config['dataset_size'] == 'sample':
        estimated_samples = 8000 * config['sample_ratio']
        if config['max_samples']:
            estimated_samples = min(estimated_samples, config['max_samples'])
        print(f"  Estimated samples: {int(estimated_samples)}")


def main():
    parser = argparse.ArgumentParser(description='Manage Airflow variables for Flickr8K pipeline')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Set up default variables')
    setup_parser.add_argument('--config', choices=list(SAMPLE_CONFIGURATIONS.keys()),
                             help='Use a predefined configuration')
    
    # Show command
    subparsers.add_parser('show', help='Show current configuration')
    
    # Test command
    subparsers.add_parser('test', help='Test DAG configuration')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export configuration to JSON')
    export_parser.add_argument('filename', help='Output JSON file')
    
    # Import command
    import_parser = subparsers.add_parser('import', help='Import configuration from JSON')
    import_parser.add_argument('filename', help='Input JSON file')
    
    # Set command
    set_parser = subparsers.add_parser('set', help='Set a specific variable')
    set_parser.add_argument('key', help='Variable key')
    set_parser.add_argument('value', help='Variable value')
    
    args = parser.parse_args()
    
    if args.command == 'setup':
        if args.config:
            setup_configuration(args.config)
        else:
            setup_default_variables()
    elif args.command == 'show':
        show_current_config()
    elif args.command == 'test':
        test_dag_config()
    elif args.command == 'export':
        export_config_to_json(args.filename)
    elif args.command == 'import':
        import_config_from_json(args.filename)
    elif args.command == 'set':
        set_airflow_variable(args.key, args.value)
    else:
        parser.print_help()


if __name__ == '__main__':
    main() 