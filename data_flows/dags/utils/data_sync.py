"""
Data synchronization utilities for the MLOps pipeline.

This module provides utilities to sync data between Airflow output
and the model training directory, ensuring consistency across the pipeline.
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging


def setup_logging() -> logging.Logger:
    """Setup logging for data sync operations."""
    logger = logging.getLogger('data_sync')
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def sync_airflow_to_model_data(
    airflow_data_path: str,
    model_data_path: str,
    symlink: bool = True,
    overwrite: bool = False
) -> Dict[str, Any]:
    """
    Sync data from Airflow output to model training directory.
    
    Args:
        airflow_data_path: Path to Airflow processed data
        model_data_path: Path to model data directory
        symlink: Whether to create symlinks instead of copying
        overwrite: Whether to overwrite existing data
    
    Returns:
        Dictionary with sync results and statistics
    """
    logger = setup_logging()
    
    airflow_path = Path(airflow_data_path)
    model_path = Path(model_data_path)
    
    logger.info(f"Starting data sync from {airflow_path} to {model_path}")
    
    results = {
        "status": "success",
        "files_synced": 0,
        "directories_created": 0,
        "errors": [],
        "warnings": []
    }
    
    try:
        # Check if airflow data exists
        processed_dir = airflow_path / "processed"
        if not processed_dir.exists():
            raise FileNotFoundError(f"Airflow processed data not found: {processed_dir}")
        
        # Create model data directory structure
        model_processed_dir = model_path / "processed"
        model_processed_dir.mkdir(parents=True, exist_ok=True)
        results["directories_created"] += 1
        
        # Files to sync
        files_to_sync = [
            "metadata.json",
            "captions.json",
            "train_data.json",
            "val_data.json",
            "test_data.json"
        ]
        
        # Sync files
        for filename in files_to_sync:
            source_file = processed_dir / filename
            target_file = model_processed_dir / filename
            
            if source_file.exists():
                if target_file.exists() and not overwrite:
                    logger.warning(f"File {filename} already exists, skipping")
                    results["warnings"].append(f"Skipped existing file: {filename}")
                    continue
                
                try:
                    if symlink and os.name != 'nt':  # Symlinks not reliable on Windows
                        if target_file.exists():
                            target_file.unlink()
                        target_file.symlink_to(source_file.absolute())
                        logger.info(f"Created symlink: {filename}")
                    else:
                        shutil.copy2(source_file, target_file)
                        logger.info(f"Copied file: {filename}")
                    
                    results["files_synced"] += 1
                    
                except Exception as e:
                    error_msg = f"Failed to sync {filename}: {str(e)}"
                    logger.error(error_msg)
                    results["errors"].append(error_msg)
            else:
                warning_msg = f"Source file not found: {filename}"
                logger.warning(warning_msg)
                results["warnings"].append(warning_msg)
        
        # Sync images directory
        source_images_dir = processed_dir / "images"
        target_images_dir = model_processed_dir / "images"
        
        if source_images_dir.exists():
            if target_images_dir.exists() and not overwrite:
                logger.warning("Images directory already exists, skipping")
                results["warnings"].append("Skipped existing images directory")
            else:
                try:
                    if target_images_dir.exists():
                        shutil.rmtree(target_images_dir)
                    
                    if symlink and os.name != 'nt':
                        target_images_dir.symlink_to(source_images_dir.absolute(), target_is_directory=True)
                        logger.info("Created symlink for images directory")
                    else:
                        shutil.copytree(source_images_dir, target_images_dir)
                        logger.info("Copied images directory")
                    
                    results["directories_created"] += 1
                    
                except Exception as e:
                    error_msg = f"Failed to sync images directory: {str(e)}"
                    logger.error(error_msg)
                    results["errors"].append(error_msg)
        else:
            warning_msg = "Source images directory not found"
            logger.warning(warning_msg)
            results["warnings"].append(warning_msg)
        
        # Update results status
        if results["errors"]:
            results["status"] = "failed"
        elif results["warnings"]:
            results["status"] = "warning"
        
        logger.info(f"Data sync complete. Status: {results['status']}")
        logger.info(f"Files synced: {results['files_synced']}, Directories created: {results['directories_created']}")
        
        return results
        
    except Exception as e:
        error_msg = f"Data sync failed: {str(e)}"
        logger.error(error_msg)
        results["status"] = "failed"
        results["errors"].append(error_msg)
        return results


def validate_data_consistency(
    airflow_data_path: str,
    model_data_path: str
) -> Dict[str, Any]:
    """
    Validate data consistency between Airflow and model directories.
    
    Args:
        airflow_data_path: Path to Airflow processed data
        model_data_path: Path to model data directory
    
    Returns:
        Dictionary with validation results
    """
    logger = setup_logging()
    
    airflow_path = Path(airflow_data_path)
    model_path = Path(model_data_path)
    
    logger.info(f"Validating data consistency between {airflow_path} and {model_path}")
    
    results = {
        "status": "success",
        "consistent": True,
        "differences": [],
        "errors": [],
        "stats": {
            "airflow": {},
            "model": {}
        }
    }
    
    try:
        # Check both processed directories exist
        airflow_processed = airflow_path / "processed"
        model_processed = model_path / "processed"
        
        if not airflow_processed.exists():
            results["errors"].append("Airflow processed directory not found")
            results["status"] = "failed"
            return results
        
        if not model_processed.exists():
            results["errors"].append("Model processed directory not found")
            results["status"] = "failed"
            return results
        
        # Compare metadata files
        airflow_metadata_file = airflow_processed / "metadata.json"
        model_metadata_file = model_processed / "metadata.json"
        
        if airflow_metadata_file.exists() and model_metadata_file.exists():
            with open(airflow_metadata_file, 'r') as f:
                airflow_metadata = json.load(f)
            with open(model_metadata_file, 'r') as f:
                model_metadata = json.load(f)
            
            results["stats"]["airflow"]["total_samples"] = airflow_metadata.get("total_samples", 0)
            results["stats"]["model"]["total_samples"] = model_metadata.get("total_samples", 0)
            
            if airflow_metadata.get("total_samples") != model_metadata.get("total_samples"):
                results["differences"].append("Different total_samples in metadata")
                results["consistent"] = False
        
        # Compare split files
        splits = ['train', 'val', 'test']
        for split in splits:
            airflow_split_file = airflow_processed / f"{split}_data.json"
            model_split_file = model_processed / f"{split}_data.json"
            
            airflow_count = 0
            model_count = 0
            
            if airflow_split_file.exists():
                with open(airflow_split_file, 'r') as f:
                    airflow_split_data = json.load(f)
                airflow_count = len(airflow_split_data)
            
            if model_split_file.exists():
                with open(model_split_file, 'r') as f:
                    model_split_data = json.load(f)
                model_count = len(model_split_data)
            
            results["stats"]["airflow"][f"{split}_samples"] = airflow_count
            results["stats"]["model"][f"{split}_samples"] = model_count
            
            if airflow_count != model_count:
                results["differences"].append(f"Different {split} sample counts: {airflow_count} vs {model_count}")
                results["consistent"] = False
        
        # Compare images directories
        airflow_images_dir = airflow_processed / "images"
        model_images_dir = model_processed / "images"
        
        if airflow_images_dir.exists():
            airflow_image_count = len(list(airflow_images_dir.glob("*.jpg"))) + len(list(airflow_images_dir.glob("*.png")))
        else:
            airflow_image_count = 0
        
        if model_images_dir.exists():
            model_image_count = len(list(model_images_dir.glob("*.jpg"))) + len(list(model_images_dir.glob("*.png")))
        else:
            model_image_count = 0
        
        results["stats"]["airflow"]["image_files"] = airflow_image_count
        results["stats"]["model"]["image_files"] = model_image_count
        
        if airflow_image_count != model_image_count:
            results["differences"].append(f"Different image file counts: {airflow_image_count} vs {model_image_count}")
            results["consistent"] = False
        
        # Set final status
        if results["differences"]:
            results["status"] = "inconsistent"
        elif results["errors"]:
            results["status"] = "failed"
        
        logger.info(f"Validation complete. Status: {results['status']}, Consistent: {results['consistent']}")
        if results["differences"]:
            logger.warning(f"Found differences: {results['differences']}")
        
        return results
        
    except Exception as e:
        error_msg = f"Validation failed: {str(e)}"
        logger.error(error_msg)
        results["status"] = "failed"
        results["errors"].append(error_msg)
        return results


def get_data_statistics(data_path: str) -> Dict[str, Any]:
    """
    Get statistics about the processed data.
    
    Args:
        data_path: Path to processed data directory
    
    Returns:
        Dictionary with data statistics
    """
    logger = setup_logging()
    data_path = Path(data_path)
    
    stats = {
        "path": str(data_path),
        "exists": data_path.exists(),
        "splits": {},
        "images": {},
        "metadata": {}
    }
    
    if not data_path.exists():
        return stats
    
    processed_dir = data_path / "processed"
    if not processed_dir.exists():
        return stats
    
    try:
        # Get split statistics
        splits = ['train', 'val', 'test']
        for split in splits:
            split_file = processed_dir / f"{split}_data.json"
            if split_file.exists():
                with open(split_file, 'r') as f:
                    split_data = json.load(f)
                stats["splits"][split] = len(split_data)
            else:
                stats["splits"][split] = 0
        
        # Get image statistics
        images_dir = processed_dir / "images"
        if images_dir.exists():
            jpg_files = list(images_dir.glob("*.jpg"))
            png_files = list(images_dir.glob("*.png"))
            stats["images"]["jpg_count"] = len(jpg_files)
            stats["images"]["png_count"] = len(png_files)
            stats["images"]["total_count"] = len(jpg_files) + len(png_files)
        else:
            stats["images"]["total_count"] = 0
        
        # Get metadata
        metadata_file = processed_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            stats["metadata"] = {
                "total_samples": metadata.get("total_samples", 0),
                "dataset_source": metadata.get("dataset_source", "unknown"),
                "download_timestamp": metadata.get("download_timestamp", "unknown"),
                "image_format": metadata.get("image_format", "unknown"),
                "image_size": metadata.get("image_size", "unknown")
            }
        
    except Exception as e:
        logger.error(f"Error getting statistics: {str(e)}")
        stats["error"] = str(e)
    
    return stats 