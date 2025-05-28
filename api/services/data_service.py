"""
Data service for handling data management and MinIO operations.
"""

import os
import sys
import uuid
import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

# Add utils for MinIO client
sys.path.append('/app/data_flows/dags/utils')

from minio_client import MinIOClient
from api_models.schemas import DatasetInfo, SampleData, StorageInfo, DownloadStatus

logger = logging.getLogger(__name__)


class DataService:
    """Service for managing data operations and MinIO integration."""
    
    def __init__(self):
        self.minio_client: Optional[MinIOClient] = None
        self.download_jobs: Dict[str, Dict[str, Any]] = {}
        
        # Initialize MinIO client
        self._init_minio_client()
    
    def _init_minio_client(self):
        """Initialize MinIO client."""
        try:
            # Get MinIO configuration from environment variables
            endpoint = os.getenv('MINIO_ENDPOINT', 'localhost:9000')
            access_key = os.getenv('MINIO_ACCESS_KEY', 'minioadmin')
            secret_key = os.getenv('MINIO_SECRET_KEY', 'minioadmin')
            secure = os.getenv('MINIO_SECURE', 'false').lower() == 'true'
            
            self.minio_client = MinIOClient(
                endpoint=endpoint,
                access_key=access_key,
                secret_key=secret_key,
                secure=secure
            )
            
            self.bucket_name = os.getenv('MINIO_BUCKET', 'image-text-data')
            
            logger.info(f"MinIO client initialized for endpoint: {endpoint}")
            
        except Exception as e:
            logger.error(f"Failed to initialize MinIO client: {e}")
            self.minio_client = None
    
    async def check_connection(self) -> bool:
        """Check MinIO connection."""
        if not self.minio_client:
            return False
        
        try:
            # Try to list buckets as a connection test
            buckets = self.minio_client.list_buckets()
            return True
            
        except Exception as e:
            logger.error(f"MinIO connection check failed: {e}")
            return False
    
    async def get_data_info(self) -> DatasetInfo:
        """Get information about available datasets."""
        try:
            # Check local data first
            local_data_path = Path("/app/model/data/processed")
            local_info = self._get_local_data_info(local_data_path)
            
            # Check MinIO data if available
            minio_info = None
            if self.minio_client:
                minio_info = await self._get_minio_data_info()
            
            # Return the most recent/complete dataset info
            if minio_info and (not local_info or minio_info.last_updated > local_info.last_updated):
                return minio_info
            elif local_info:
                return local_info
            else:
                # Return default empty dataset info
                return DatasetInfo(
                    name="flickr8k",
                    total_samples=0,
                    splits={"train": 0, "val": 0, "test": 0},
                    last_updated=datetime.now(),
                    source="none",
                    size_mb=0.0
                )
                
        except Exception as e:
            logger.error(f"Failed to get data info: {e}")
            raise
    
    def _get_local_data_info(self, data_path: Path) -> Optional[DatasetInfo]:
        """Get information about local dataset."""
        try:
            if not data_path.exists():
                return None
            
            # Check for metadata file
            metadata_file = data_path / "metadata.json"
            if not metadata_file.exists():
                return None
            
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Get split information
            splits = {}
            for split in ['train', 'val', 'test']:
                split_file = data_path / f"{split}_data.json"
                if split_file.exists():
                    with open(split_file, 'r') as f:
                        split_data = json.load(f)
                    splits[split] = len(split_data)
                else:
                    splits[split] = 0
            
            # Calculate total size
            total_size = sum(
                f.stat().st_size for f in data_path.rglob('*') if f.is_file()
            )
            
            return DatasetInfo(
                name="flickr8k",
                total_samples=metadata.get("total_samples", 0),
                splits=splits,
                last_updated=datetime.fromisoformat(metadata.get("download_timestamp", datetime.now().isoformat())),
                source="local",
                size_mb=round(total_size / (1024 * 1024), 2)
            )
            
        except Exception as e:
            logger.warning(f"Failed to get local data info: {e}")
            return None
    
    async def _get_minio_data_info(self) -> Optional[DatasetInfo]:
        """Get information about dataset in MinIO."""
        try:
            if not self.minio_client:
                return None
            
            # Check if metadata exists
            metadata_path = "datasets/flickr8k/metadata.json"
            if not self.minio_client.object_exists(self.bucket_name, metadata_path):
                return None
            
            # Download metadata
            metadata = self.minio_client.download_json(self.bucket_name, metadata_path)
            
            # Get object list to calculate size and splits
            objects = self.minio_client.list_objects(
                bucket_name=self.bucket_name,
                prefix="datasets/flickr8k/"
            )
            
            total_size = sum(obj.size for obj in objects)
            
            # Count split files
            splits = {"train": 0, "val": 0, "test": 0}
            
            for obj in objects:
                if obj.object_name.endswith('_data.json'):
                    split_name = obj.object_name.split('/')[-1].replace('_data.json', '')
                    if split_name in splits:
                        # Download and count samples
                        try:
                            split_data = self.minio_client.download_json(self.bucket_name, obj.object_name)
                            splits[split_name] = len(split_data) if isinstance(split_data, list) else 0
                        except:
                            pass
            
            return DatasetInfo(
                name="flickr8k",
                total_samples=metadata.get("total_samples", 0),
                splits=splits,
                last_updated=datetime.fromisoformat(metadata.get("download_timestamp", datetime.now().isoformat())),
                source="minio",
                size_mb=round(total_size / (1024 * 1024), 2)
            )
            
        except Exception as e:
            logger.warning(f"Failed to get MinIO data info: {e}")
            return None
    
    async def download_latest_dataset(self, background_tasks=None) -> str:
        """Download latest dataset from MinIO."""
        if not self.minio_client:
            raise RuntimeError("MinIO client not available")
        
        # Generate download ID
        download_id = str(uuid.uuid4())
        
        # Initialize download job
        self.download_jobs[download_id] = {
            "download_id": download_id,
            "status": "starting",
            "progress": 0.0,
            "files_downloaded": 0,
            "total_files": 0,
            "started_at": datetime.now(),
            "completed_at": None
        }
        
        # Start download in background
        if background_tasks:
            background_tasks.add_task(self._run_download, download_id)
        else:
            # Start in asyncio task
            asyncio.create_task(self._run_download(download_id))
        
        logger.info(f"Dataset download started with ID: {download_id}")
        
        return download_id
    
    async def _run_download(self, download_id: str):
        """Run the actual download process."""
        try:
            logger.info(f"Starting dataset download for ID: {download_id}")
            
            job = self.download_jobs[download_id]
            job["status"] = "running"
            
            # Get list of objects to download
            objects = self.minio_client.list_objects(
                bucket_name=self.bucket_name,
                prefix="datasets/flickr8k/"
            )
            
            object_list = list(objects)
            job["total_files"] = len(object_list)
            
            # Create local directory
            local_data_path = Path("/app/data/processed")
            local_data_path.mkdir(parents=True, exist_ok=True)
            
            # Download files
            downloaded_count = 0
            
            for obj in object_list:
                try:
                    # Calculate local path
                    relative_path = obj.object_name.replace("datasets/flickr8k/", "")
                    local_path = local_data_path / relative_path
                    
                    # Create parent directory
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Download file
                    self.minio_client.download_file(
                        bucket_name=self.bucket_name,
                        object_name=obj.object_name,
                        file_path=str(local_path)
                    )
                    
                    downloaded_count += 1
                    job["files_downloaded"] = downloaded_count
                    job["progress"] = (downloaded_count / len(object_list)) * 100
                    
                    logger.debug(f"Downloaded {relative_path}")
                    
                except Exception as e:
                    logger.error(f"Failed to download {obj.object_name}: {e}")
                    continue
            
            # Mark as completed
            job["status"] = "completed"
            job["progress"] = 100.0
            job["completed_at"] = datetime.now()
            
            logger.info(f"Dataset download {download_id} completed: {downloaded_count}/{len(object_list)} files")
            
        except Exception as e:
            # Mark as failed
            self.download_jobs[download_id]["status"] = "failed"
            self.download_jobs[download_id]["error"] = str(e)
            
            logger.error(f"Dataset download {download_id} failed: {e}")
    
    async def get_download_status(self, download_id: str) -> DownloadStatus:
        """Get download status."""
        if download_id not in self.download_jobs:
            raise ValueError(f"Download ID {download_id} not found")
        
        job = self.download_jobs[download_id]
        
        return DownloadStatus(**job)
    
    async def get_sample_data(self, limit: int = 10) -> List[SampleData]:
        """Get sample data points from the dataset."""
        try:
            # Try local data first
            local_data_path = Path("/app/model/data/processed")
            
            # Check different possible paths
            possible_paths = [
                local_data_path,
                Path("/app/data/processed"),
                Path("./data/processed")
            ]
            
            for data_path in possible_paths:
                if data_path.exists():
                    return self._get_local_sample_data(data_path, limit)
            
            # If no local data, try MinIO
            if self.minio_client:
                return await self._get_minio_sample_data(limit)
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to get sample data: {e}")
            return []
    
    def _get_local_sample_data(self, data_path: Path, limit: int) -> List[SampleData]:
        """Get sample data from local files."""
        samples = []
        
        try:
            # Try to get samples from train split first
            for split in ['train', 'val', 'test']:
                split_file = data_path / f"{split}_data.json"
                
                if split_file.exists() and len(samples) < limit:
                    with open(split_file, 'r') as f:
                        split_data = json.load(f)
                    
                    # Take samples from this split
                    remaining = limit - len(samples)
                    split_samples = split_data[:remaining]
                    
                    for sample in split_samples:
                        # Construct full image path
                        image_path = data_path / "images" / sample.get("image_file", "")
                        
                        samples.append(SampleData(
                            image_path=str(image_path),
                            caption=sample.get("caption", ""),
                            image_id=sample.get("image_id", ""),
                            split=split
                        ))
                    
                    if len(samples) >= limit:
                        break
            
        except Exception as e:
            logger.error(f"Failed to get local sample data: {e}")
        
        return samples
    
    async def _get_minio_sample_data(self, limit: int) -> List[SampleData]:
        """Get sample data from MinIO."""
        samples = []
        
        try:
            # Download sample from train split
            train_data_path = "datasets/flickr8k/train_data.json"
            
            if self.minio_client.object_exists(self.bucket_name, train_data_path):
                train_data = self.minio_client.download_json(self.bucket_name, train_data_path)
                
                # Take limited samples
                sample_data = train_data[:limit]
                
                for sample in sample_data:
                    # Note: For MinIO samples, we return the MinIO path
                    # In practice, you'd want to provide presigned URLs or download the images
                    image_file = sample.get("image_file", "")
                    minio_image_path = f"minio://{self.bucket_name}/datasets/flickr8k/images/{image_file}"
                    
                    samples.append(SampleData(
                        image_path=minio_image_path,
                        caption=sample.get("caption", ""),
                        image_id=sample.get("image_id", ""),
                        split="train"
                    ))
        
        except Exception as e:
            logger.error(f"Failed to get MinIO sample data: {e}")
        
        return samples
    
    async def get_sample_images(self, limit: int = 20) -> List[str]:
        """Get list of sample image paths."""
        try:
            # Get more sample data than needed to account for duplicates
            # Since each image has ~5 captions, we need more samples to get unique images
            extended_limit = limit * 10  # Get more samples to ensure we have enough unique images
            samples = await self.get_sample_data(extended_limit)
            
            # Extract unique image paths (filter duplicates)
            seen_paths = set()
            unique_image_paths = []
            
            for sample in samples:
                if (not sample.image_path.startswith("minio://") and 
                    Path(sample.image_path).exists() and 
                    sample.image_path not in seen_paths):
                    seen_paths.add(sample.image_path)
                    unique_image_paths.append(sample.image_path)
                    
                    # Stop once we have enough unique images
                    if len(unique_image_paths) >= limit:
                        break
            
            return unique_image_paths
            
        except Exception as e:
            logger.error(f"Failed to get sample images: {e}")
            return []
    
    async def get_storage_info(self) -> StorageInfo:
        """Get storage system information."""
        try:
            if not self.minio_client:
                return StorageInfo(
                    connected=False,
                    endpoint="not_configured",
                    bucket_name="",
                    total_objects=0,
                    total_size_mb=0.0,
                    last_sync=None
                )
            
            # Get bucket info
            objects = self.minio_client.list_objects(
                bucket_name=self.bucket_name,
                prefix="datasets/flickr8k/"
            )
            
            object_list = list(objects)
            total_size = sum(obj.size for obj in object_list)
            
            # Get last sync time from metadata if available
            last_sync = None
            try:
                upload_metadata_path = "datasets/flickr8k/upload_metadata.json"
                if self.minio_client.object_exists(self.bucket_name, upload_metadata_path):
                    upload_metadata = self.minio_client.download_json(self.bucket_name, upload_metadata_path)
                    last_sync = datetime.fromisoformat(upload_metadata.get("upload_timestamp", ""))
            except:
                pass
            
            return StorageInfo(
                connected=True,
                endpoint=self.minio_client.endpoint,
                bucket_name=self.bucket_name,
                total_objects=len(object_list),
                total_size_mb=round(total_size / (1024 * 1024), 2),
                last_sync=last_sync
            )
            
        except Exception as e:
            logger.error(f"Failed to get storage info: {e}")
            return StorageInfo(
                connected=False,
                endpoint="error",
                bucket_name="",
                total_objects=0,
                total_size_mb=0.0,
                last_sync=None
            ) 