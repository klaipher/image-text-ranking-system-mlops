"""
MinIO client utility for object storage operations.

This module provides a convenient interface for interacting with MinIO
object storage, including uploading/downloading files and managing buckets.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from minio import Minio
from minio.error import S3Error


def setup_logging() -> logging.Logger:
    """Setup logging for MinIO operations."""
    logger = logging.getLogger('minio_client')
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


class MinIOClient:
    """MinIO client for object storage operations."""
    
    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        secure: bool = False
    ):
        """
        Initialize MinIO client.
        
        Args:
            endpoint: MinIO server endpoint (e.g., 'localhost:9000')
            access_key: Access key for authentication
            secret_key: Secret key for authentication
            secure: Use HTTPS if True, HTTP if False
        """
        self.endpoint = endpoint
        self.access_key = access_key
        self.secret_key = secret_key
        self.secure = secure
        
        self.logger = setup_logging()
        
        # Initialize MinIO client
        self.client = Minio(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )
        
        self.logger.info(f"MinIO client initialized for endpoint: {endpoint}")
    
    def create_bucket(self, bucket_name: str) -> bool:
        """
        Create a bucket if it doesn't exist.
        
        Args:
            bucket_name: Name of the bucket to create
            
        Returns:
            True if bucket was created, False if it already existed
        """
        try:
            if self.client.bucket_exists(bucket_name):
                self.logger.info(f"Bucket '{bucket_name}' already exists")
                return False
            
            self.client.make_bucket(bucket_name)
            self.logger.info(f"Created bucket: {bucket_name}")
            return True
            
        except S3Error as e:
            self.logger.error(f"Failed to create bucket '{bucket_name}': {e}")
            raise
    
    def list_buckets(self) -> List[str]:
        """
        List all buckets.
        
        Returns:
            List of bucket names
        """
        try:
            buckets = self.client.list_buckets()
            bucket_names = [bucket.name for bucket in buckets]
            return bucket_names
            
        except S3Error as e:
            self.logger.error(f"Failed to list buckets: {e}")
            raise
    
    def upload_file(
        self,
        bucket_name: str,
        object_name: str,
        file_path: str,
        content_type: str = None
    ) -> Dict[str, Any]:
        """
        Upload a file to MinIO.
        
        Args:
            bucket_name: Name of the bucket
            object_name: Name of the object in the bucket
            file_path: Local path to the file
            content_type: MIME type of the file
            
        Returns:
            Upload result information
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Auto-detect content type if not provided
            if content_type is None:
                if file_path.suffix.lower() in ['.jpg', '.jpeg']:
                    content_type = 'image/jpeg'
                elif file_path.suffix.lower() == '.png':
                    content_type = 'image/png'
                elif file_path.suffix.lower() == '.json':
                    content_type = 'application/json'
                else:
                    content_type = 'application/octet-stream'
            
            # Upload file
            result = self.client.fput_object(
                bucket_name=bucket_name,
                object_name=object_name,
                file_path=str(file_path),
                content_type=content_type
            )
            
            file_size = file_path.stat().st_size
            
            self.logger.info(f"Uploaded {file_path.name} to {bucket_name}/{object_name}")
            
            return {
                "status": "success",
                "bucket_name": bucket_name,
                "object_name": object_name,
                "file_size": file_size,
                "etag": result.etag
            }
            
        except Exception as e:
            self.logger.error(f"Failed to upload {file_path}: {e}")
            raise
    
    def upload_json(
        self,
        bucket_name: str,
        object_name: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Upload JSON data to MinIO.
        
        Args:
            bucket_name: Name of the bucket
            object_name: Name of the object in the bucket
            data: Dictionary to upload as JSON
            
        Returns:
            Upload result information
        """
        try:
            import io
            
            # Convert data to JSON bytes
            json_data = json.dumps(data, indent=2).encode('utf-8')
            json_io = io.BytesIO(json_data)
            
            # Upload
            result = self.client.put_object(
                bucket_name=bucket_name,
                object_name=object_name,
                data=json_io,
                length=len(json_data),
                content_type='application/json'
            )
            
            self.logger.info(f"Uploaded JSON data to {bucket_name}/{object_name}")
            
            return {
                "status": "success",
                "bucket_name": bucket_name,
                "object_name": object_name,
                "file_size": len(json_data),
                "etag": result.etag
            }
            
        except Exception as e:
            self.logger.error(f"Failed to upload JSON to {object_name}: {e}")
            raise
    
    def upload_directory(
        self,
        bucket_name: str,
        local_directory: str,
        object_prefix: str = ""
    ) -> Dict[str, Any]:
        """
        Upload a directory recursively to MinIO.
        
        Args:
            bucket_name: Name of the bucket
            local_directory: Local directory path
            object_prefix: Prefix for object names in the bucket
            
        Returns:
            Upload result summary
        """
        try:
            local_path = Path(local_directory)
            
            if not local_path.exists():
                raise FileNotFoundError(f"Directory not found: {local_path}")
            
            uploaded_files = []
            total_size = 0
            errors = []
            
            # Walk through directory
            for file_path in local_path.rglob('*'):
                if file_path.is_file():
                    # Calculate relative path
                    relative_path = file_path.relative_to(local_path)
                    object_name = object_prefix + str(relative_path).replace('\\', '/')
                    
                    try:
                        result = self.upload_file(
                            bucket_name=bucket_name,
                            object_name=object_name,
                            file_path=str(file_path)
                        )
                        
                        uploaded_files.append({
                            "local_path": str(file_path),
                            "object_name": object_name,
                            "size": result["file_size"]
                        })
                        
                        total_size += result["file_size"]
                        
                    except Exception as e:
                        error_msg = f"Failed to upload {file_path}: {str(e)}"
                        self.logger.error(error_msg)
                        errors.append(error_msg)
            
            upload_results = {
                "status": "success" if not errors else "partial_success",
                "files_uploaded": len(uploaded_files),
                "total_size": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "errors": errors,
                "uploaded_files": uploaded_files
            }
            
            self.logger.info(f"Directory upload completed: {len(uploaded_files)} files, {upload_results['total_size_mb']} MB")
            
            if errors:
                self.logger.warning(f"Upload completed with {len(errors)} errors")
            
            return upload_results
            
        except Exception as e:
            self.logger.error(f"Failed to upload directory {local_directory}: {e}")
            raise
    
    def download_file(
        self,
        bucket_name: str,
        object_name: str,
        file_path: str
    ) -> Dict[str, Any]:
        """
        Download a file from MinIO.
        
        Args:
            bucket_name: Name of the bucket
            object_name: Name of the object in the bucket
            file_path: Local path to save the file
            
        Returns:
            Download result information
        """
        try:
            # Ensure directory exists
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Download file
            self.client.fget_object(
                bucket_name=bucket_name,
                object_name=object_name,
                file_path=str(file_path)
            )
            
            file_size = file_path.stat().st_size
            
            self.logger.info(f"Downloaded {bucket_name}/{object_name} to {file_path}")
            
            return {
                "status": "success",
                "bucket_name": bucket_name,
                "object_name": object_name,
                "local_path": str(file_path),
                "file_size": file_size
            }
            
        except Exception as e:
            self.logger.error(f"Failed to download {bucket_name}/{object_name}: {e}")
            raise
    
    def download_json(
        self,
        bucket_name: str,
        object_name: str
    ) -> Dict[str, Any]:
        """
        Download and parse JSON data from MinIO.
        
        Args:
            bucket_name: Name of the bucket
            object_name: Name of the object in the bucket
            
        Returns:
            Parsed JSON data
        """
        try:
            response = self.client.get_object(bucket_name, object_name)
            data = json.loads(response.read().decode('utf-8'))
            response.close()
            response.release_conn()
            
            self.logger.info(f"Downloaded JSON from {bucket_name}/{object_name}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to download JSON {bucket_name}/{object_name}: {e}")
            raise
    
    def list_objects(
        self,
        bucket_name: str,
        prefix: str = None,
        recursive: bool = True
    ) -> List[Any]:
        """
        List objects in a bucket.
        
        Args:
            bucket_name: Name of the bucket
            prefix: Prefix to filter objects
            recursive: List objects recursively
            
        Returns:
            List of object information
        """
        try:
            objects = self.client.list_objects(
                bucket_name=bucket_name,
                prefix=prefix,
                recursive=recursive
            )
            
            object_list = list(objects)
            self.logger.info(f"Listed {len(object_list)} objects from {bucket_name}")
            
            return object_list
            
        except Exception as e:
            self.logger.error(f"Failed to list objects in {bucket_name}: {e}")
            raise
    
    def delete_object(
        self,
        bucket_name: str,
        object_name: str
    ) -> Dict[str, Any]:
        """
        Delete an object from MinIO.
        
        Args:
            bucket_name: Name of the bucket
            object_name: Name of the object to delete
            
        Returns:
            Deletion result information
        """
        try:
            self.client.remove_object(bucket_name, object_name)
            
            self.logger.info(f"Deleted {bucket_name}/{object_name}")
            
            return {
                "status": "success",
                "bucket_name": bucket_name,
                "object_name": object_name
            }
            
        except Exception as e:
            self.logger.error(f"Failed to delete {bucket_name}/{object_name}: {e}")
            raise
    
    def object_exists(
        self,
        bucket_name: str,
        object_name: str
    ) -> bool:
        """
        Check if an object exists in MinIO.
        
        Args:
            bucket_name: Name of the bucket
            object_name: Name of the object
            
        Returns:
            True if object exists, False otherwise
        """
        try:
            self.client.stat_object(bucket_name, object_name)
            return True
        except S3Error as e:
            if e.code == 'NoSuchKey':
                return False
            else:
                raise
    
    def get_object_info(
        self,
        bucket_name: str,
        object_name: str
    ) -> Dict[str, Any]:
        """
        Get object information from MinIO.
        
        Args:
            bucket_name: Name of the bucket
            object_name: Name of the object
            
        Returns:
            Object information
        """
        try:
            stat = self.client.stat_object(bucket_name, object_name)
            
            return {
                "bucket_name": bucket_name,
                "object_name": object_name,
                "size": stat.size,
                "etag": stat.etag,
                "last_modified": stat.last_modified,
                "content_type": stat.content_type
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get info for {bucket_name}/{object_name}: {e}")
            raise 