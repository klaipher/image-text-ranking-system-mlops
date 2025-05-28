"""
Pydantic schemas for API request and response models.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


# Request Models
class TextToImageRequest(BaseModel):
    """Request model for text-to-image ranking."""
    text_query: str = Field(..., description="Text query for image search")
    image_paths: List[str] = Field(..., description="List of image file paths to rank")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of top results to return")


class ImageToTextRequest(BaseModel):
    """Request model for image-to-text ranking."""
    image_path: str = Field(..., description="Path to query image")
    texts: List[str] = Field(..., description="List of text candidates to rank")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of top results to return")


class BatchRankingRequest(BaseModel):
    """Request model for batch ranking."""
    text_queries: List[str] = Field(..., description="List of text queries")
    image_paths: List[str] = Field(..., description="List of image file paths")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of top results per query")


class TrainingRequest(BaseModel):
    """Request model for starting model training."""
    data_source: str = Field(default="minio", description="Data source (minio, local)")
    epochs: int = Field(default=10, ge=1, le=100, description="Number of training epochs")
    batch_size: int = Field(default=32, ge=1, le=512, description="Training batch size")
    learning_rate: float = Field(default=1e-4, gt=0, lt=1, description="Learning rate")
    use_existing_data: bool = Field(default=True, description="Use existing downloaded data")
    experiment_name: str = Field(default="api_training", description="MLflow experiment name")
    run_name: Optional[str] = Field(default=None, description="MLflow run name")
    save_checkpoints: bool = Field(default=True, description="Save training checkpoints")

    @validator('run_name', pre=True, always=True)
    def set_run_name(cls, v, values):
        if v is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"api_training_{timestamp}"
        return v


# Response Models
class RankingResult(BaseModel):
    """Single ranking result."""
    rank: int = Field(..., description="Rank position (1-based)")
    score: float = Field(..., description="Similarity score")
    image_path: Optional[str] = Field(None, description="Image path (for text-to-image)")
    text: Optional[str] = Field(None, description="Text (for image-to-text)")


class ModelInfo(BaseModel):
    """Model information."""
    model_name: str = Field(..., description="Model name")
    model_path: str = Field(..., description="Path to model file")
    device: str = Field(..., description="Device (cpu, mps, cuda)")
    embedding_dim: int = Field(..., description="Embedding dimension")
    vocab_size: int = Field(..., description="Vocabulary size")
    loaded_at: datetime = Field(..., description="When model was loaded")
    model_size_mb: float = Field(..., description="Model size in MB")


class TrainingStatus(BaseModel):
    """Training status information."""
    is_active: bool = Field(..., description="Whether training is currently active")
    training_id: Optional[str] = Field(None, description="Current training job ID")
    status: str = Field(..., description="Training status (not_started, running, completed, failed)")
    current_epoch: Optional[int] = Field(None, description="Current epoch number")
    total_epochs: Optional[int] = Field(None, description="Total epochs configured")
    current_loss: Optional[float] = Field(None, description="Current training loss")
    best_score: Optional[float] = Field(None, description="Best validation score")
    elapsed_time: Optional[float] = Field(None, description="Elapsed training time in seconds")
    estimated_remaining: Optional[float] = Field(None, description="Estimated remaining time in seconds")
    last_update: Optional[datetime] = Field(None, description="Last status update")


class HealthCheck(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    message: str = Field(..., description="Status message")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    training_active: bool = Field(..., description="Whether training is active")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")


# Data Models
class DatasetInfo(BaseModel):
    """Dataset information."""
    name: str = Field(..., description="Dataset name")
    total_samples: int = Field(..., description="Total number of samples")
    splits: Dict[str, int] = Field(..., description="Data split information")
    last_updated: datetime = Field(..., description="Last update timestamp")
    source: str = Field(..., description="Data source (minio, local)")
    size_mb: float = Field(..., description="Dataset size in MB")


class SampleData(BaseModel):
    """Sample data point."""
    image_path: str = Field(..., description="Path to image file")
    caption: str = Field(..., description="Image caption")
    image_id: str = Field(..., description="Unique image identifier")
    split: str = Field(..., description="Data split (train, val, test)")


# Training History Models
class TrainingRun(BaseModel):
    """Training run information."""
    run_id: str = Field(..., description="Training run ID")
    experiment_name: str = Field(..., description="MLflow experiment name")
    run_name: str = Field(..., description="MLflow run name")
    status: str = Field(..., description="Run status")
    start_time: datetime = Field(..., description="Training start time")
    end_time: Optional[datetime] = Field(None, description="Training end time")
    duration: Optional[float] = Field(None, description="Training duration in seconds")
    final_loss: Optional[float] = Field(None, description="Final training loss")
    best_score: Optional[float] = Field(None, description="Best validation score")
    config: Dict[str, Any] = Field(..., description="Training configuration")


class TrainingHistory(BaseModel):
    """Training history."""
    total_runs: int = Field(..., description="Total number of training runs")
    runs: List[TrainingRun] = Field(..., description="List of training runs")
    last_successful_run: Optional[TrainingRun] = Field(None, description="Last successful run")


# Error Models
class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")


# File Upload Models
class UploadResponse(BaseModel):
    """File upload response."""
    filename: str = Field(..., description="Uploaded filename")
    size: int = Field(..., description="File size in bytes")
    content_type: str = Field(..., description="File content type")
    upload_path: str = Field(..., description="Path where file was saved")
    upload_time: datetime = Field(default_factory=datetime.now, description="Upload timestamp")


# Demo Models
class DemoQuery(BaseModel):
    """Demo query model."""
    text: str = Field(..., description="Demo text query")
    description: str = Field(..., description="Query description")
    expected_results: Optional[List[str]] = Field(None, description="Expected result types")


class DemoResponse(BaseModel):
    """Demo response model."""
    query: str = Field(..., description="Query text")
    results: List[RankingResult] = Field(..., description="Ranking results")
    total_images: int = Field(..., description="Total images searched")
    demo: bool = Field(default=True, description="Indicates this is a demo response")
    execution_time: float = Field(..., description="Query execution time in seconds")


# MinIO/Storage Models
class StorageInfo(BaseModel):
    """Storage system information."""
    connected: bool = Field(..., description="Whether storage is connected")
    endpoint: str = Field(..., description="Storage endpoint")
    bucket_name: str = Field(..., description="Bucket name")
    total_objects: int = Field(..., description="Total objects in bucket")
    total_size_mb: float = Field(..., description="Total size in MB")
    last_sync: Optional[datetime] = Field(None, description="Last synchronization time")


class DownloadStatus(BaseModel):
    """Data download status."""
    download_id: str = Field(..., description="Download job ID")
    status: str = Field(..., description="Download status")
    progress: float = Field(..., ge=0, le=100, description="Download progress percentage")
    files_downloaded: int = Field(..., description="Number of files downloaded")
    total_files: int = Field(..., description="Total files to download")
    download_speed_mbps: Optional[float] = Field(None, description="Download speed in MB/s")
    estimated_remaining: Optional[float] = Field(None, description="Estimated remaining time in seconds")
    started_at: datetime = Field(..., description="Download start time")
    completed_at: Optional[datetime] = Field(None, description="Download completion time")
