"""
Configuration module for the image-text ranking model.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass
class ModelConfig:
    """Configuration for the dual-encoder model."""
    
    # Model architecture
    image_encoder: str = "mobilenet_v3_small"
    text_encoder_dim: int = 512
    embedding_dim: int = 256
    vocab_size: int = 10000
    max_text_length: int = 64
    dropout_rate: float = 0.1
    
    # Image processing
    image_size: Tuple[int, int] = (224, 224)
    image_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    
    # Training
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 10
    warmup_steps: int = 500
    weight_decay: float = 0.01
    temperature: float = 0.07  # For contrastive learning
    
    # Evaluation
    recall_at_k: Tuple[int, ...] = (1, 5, 10)
    
    # Hardware
    device: str = "auto"  # Will be set in __post_init__
    
    def __post_init__(self):
        """Set device after torch import."""
        import torch
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"


@dataclass
class DataConfig:
    """Configuration for data processing and paths."""
    
    # Paths
    data_root: Path = Path("data")
    raw_data_path: Path = Path("data/raw")
    processed_data_path: Path = Path("data/processed")
    models_path: Path = Path("models")
    logs_path: Path = Path("logs")
    
    # Dataset
    dataset_name: str = "flickr8k"
    train_split: float = 0.6
    val_split: float = 0.2
    test_split: float = 0.2
    captions_per_image: int = 5
    
    # Data processing
    min_caption_length: int = 3
    max_caption_length: int = 50
    
    # Storage (for future integration)
    use_object_storage: bool = False
    storage_endpoint: str = "http://localhost:9000"
    storage_bucket: str = "image-text-data"
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        for path in [self.data_root, self.raw_data_path, self.processed_data_path, 
                     self.models_path, self.logs_path]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class TrainingConfig:
    """Configuration for training process."""
    
    # Experiment tracking
    experiment_name: str = "image_text_ranking_baseline"
    run_name: str = "baseline_run"
    tracking_uri: str = "file:./mlruns"
    
    # Model saving
    save_every_n_epochs: int = 2
    save_best_model: bool = True
    early_stopping_patience: int = 3
    
    # Logging
    log_every_n_steps: int = 100
    eval_every_n_epochs: int = 1
    
    # Reproducibility
    random_seed: int = 42


# Global configuration instances
model_config = ModelConfig()
data_config = DataConfig()
training_config = TrainingConfig()


def get_config():
    """Get all configuration objects."""
    return {
        "model": model_config,
        "data": data_config,
        "training": training_config
    } 