"""
Enhanced Airflow DAG for Flickr8K dataset with model data synchronization.

This DAG extends the basic data pipeline to include synchronization with
the model training directory, ensuring data consistency across the MLOps pipeline.
"""

import os
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable

# Import the base DAG functions
from flickr8k_data_pipeline import (
    get_dag_config,
    create_output_directories,
    download_flickr8k_dataset,
    create_data_splits,
    validate_dataset,
    upload_to_object_storage
)

# Import sync utilities
from utils.data_sync import (
    sync_airflow_to_model_data,
    validate_data_consistency,
    get_data_statistics
)


# Default arguments for the DAG
default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Enhanced DAG definition
dag = DAG(
    'flickr8k_with_model_sync',
    default_args=default_args,
    description='Flickr8K dataset pipeline with model data synchronization',
    schedule=None,  # Manual trigger only
    max_active_runs=1,
    catchup=False,
    tags=['data', 'flickr8k', 'image-text-ranking', 'sync'],
)


def sync_to_model_directory(**context) -> Dict[str, Any]:
    """Sync processed data to model training directory."""
    config = get_dag_config(**context)
    
    # Get paths
    airflow_data_path = config['data_output_path']
    model_data_path = Variable.get('model_data_path', default='../model/data')
    
    # Sync configuration
    sync_config = {
        'symlink': config.get('use_symlinks', True),
        'overwrite': config.get('overwrite_model_data', False)
    }
    
    print(f"Syncing data from {airflow_data_path} to {model_data_path}")
    print(f"Sync configuration: {sync_config}")
    
    # Perform sync
    sync_results = sync_airflow_to_model_data(
        airflow_data_path=airflow_data_path,
        model_data_path=model_data_path,
        symlink=sync_config['symlink'],
        overwrite=sync_config['overwrite']
    )
    
    print(f"Sync completed with status: {sync_results['status']}")
    print(f"Files synced: {sync_results['files_synced']}")
    print(f"Directories created: {sync_results['directories_created']}")
    
    if sync_results['errors']:
        print(f"Errors: {sync_results['errors']}")
    if sync_results['warnings']:
        print(f"Warnings: {sync_results['warnings']}")
    
    return sync_results


def validate_sync_consistency(**context) -> Dict[str, Any]:
    """Validate data consistency between Airflow and model directories."""
    config = get_dag_config(**context)
    
    airflow_data_path = config['data_output_path']
    model_data_path = Variable.get('model_data_path', default='../model/data')
    
    print(f"Validating consistency between {airflow_data_path} and {model_data_path}")
    
    # Validate consistency
    validation_results = validate_data_consistency(
        airflow_data_path=airflow_data_path,
        model_data_path=model_data_path
    )
    
    print(f"Validation completed with status: {validation_results['status']}")
    print(f"Data consistent: {validation_results['consistent']}")
    
    if validation_results['differences']:
        print(f"Differences found: {validation_results['differences']}")
    if validation_results['errors']:
        print(f"Errors: {validation_results['errors']}")
    
    # Print statistics
    print("\nData Statistics:")
    print(f"Airflow data: {validation_results['stats']['airflow']}")
    print(f"Model data: {validation_results['stats']['model']}")
    
    return validation_results


def generate_data_report(**context) -> Dict[str, Any]:
    """Generate a comprehensive data report."""
    config = get_dag_config(**context)
    
    airflow_data_path = config['data_output_path']
    model_data_path = Variable.get('model_data_path', default='../model/data')
    
    print("Generating comprehensive data report...")
    
    # Get statistics for both directories
    airflow_stats = get_data_statistics(airflow_data_path)
    model_stats = get_data_statistics(model_data_path)
    
    # Create report
    report = {
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'airflow_data': airflow_stats,
        'model_data': model_stats,
        'pipeline_summary': {
            'total_samples_processed': airflow_stats.get('metadata', {}).get('total_samples', 0),
            'data_splits': airflow_stats.get('splits', {}),
            'image_count': airflow_stats.get('images', {}).get('total_count', 0),
            'sync_status': 'success' if airflow_stats.get('exists') and model_stats.get('exists') else 'incomplete'
        }
    }
    
    # Save report
    report_path = Path(airflow_data_path) / "data_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Data report saved to: {report_path}")
    print(f"Pipeline summary: {report['pipeline_summary']}")
    
    return report


def cleanup_temp_files(**context) -> Dict[str, Any]:
    """Clean up temporary files and optimize storage."""
    config = get_dag_config(**context)
    
    cleanup_results = {
        'status': 'success',
        'files_removed': 0,
        'space_freed': 0,
        'errors': []
    }
    
    try:
        airflow_data_path = Path(config['data_output_path'])
        
        # Clean up raw directory if processing is complete
        raw_dir = airflow_data_path / "raw"
        if raw_dir.exists() and config.get('cleanup_raw_data', False):
            import shutil
            shutil.rmtree(raw_dir)
            cleanup_results['files_removed'] += 1
            print("Removed raw data directory")
        
        # Clean up logs older than 7 days
        logs_dir = airflow_data_path / "logs"
        if logs_dir.exists():
            cutoff_date = datetime.now() - timedelta(days=7)
            for log_file in logs_dir.glob("*.log"):
                if datetime.fromtimestamp(log_file.stat().st_mtime) < cutoff_date:
                    log_file.unlink()
                    cleanup_results['files_removed'] += 1
        
        print(f"Cleanup completed. Files removed: {cleanup_results['files_removed']}")
        
    except Exception as e:
        cleanup_results['status'] = 'failed'
        cleanup_results['errors'].append(str(e))
        print(f"Cleanup failed: {str(e)}")
    
    return cleanup_results


# Define tasks
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

validate_airflow_data_task = PythonOperator(
    task_id='validate_airflow_dataset',
    python_callable=validate_dataset,
    dag=dag,
)

sync_to_model_task = PythonOperator(
    task_id='sync_to_model_directory',
    python_callable=sync_to_model_directory,
    dag=dag,
)

validate_sync_task = PythonOperator(
    task_id='validate_sync_consistency',
    python_callable=validate_sync_consistency,
    dag=dag,
)

generate_report_task = PythonOperator(
    task_id='generate_data_report',
    python_callable=generate_data_report,
    dag=dag,
)

upload_task = PythonOperator(
    task_id='upload_to_object_storage',
    python_callable=upload_to_object_storage,
    dag=dag,
)

cleanup_task = PythonOperator(
    task_id='cleanup_temp_files',
    python_callable=cleanup_temp_files,
    dag=dag,
)

# Set task dependencies
create_dirs_task >> download_task >> create_splits_task >> validate_airflow_data_task
validate_airflow_data_task >> sync_to_model_task >> validate_sync_task
validate_sync_task >> generate_report_task
generate_report_task >> [upload_task, cleanup_task] 