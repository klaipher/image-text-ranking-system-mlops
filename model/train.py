#!/usr/bin/env python3
"""
Main training script for the image-text ranking baseline model.
"""

import argparse
import torch
import random
import numpy as np
from pathlib import Path

from src.config import model_config, training_config, data_config
from src.training.trainer import Trainer
from src.data import get_dataloaders


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser(description="Train image-text ranking model")
    parser.add_argument("--data-dir", type=str, help="Path to data directory")
    parser.add_argument("--epochs", type=int, default=model_config.num_epochs, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=model_config.batch_size, help="Batch size")
    parser.add_argument("--lr", type=float, default=model_config.learning_rate, help="Learning rate")
    parser.add_argument("--device", type=str, choices=['cpu', 'mps', 'cuda'], help="Device to use")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--experiment-name", type=str, default=training_config.experiment_name, 
                       help="MLflow experiment name")
    parser.add_argument("--run-name", type=str, default=training_config.run_name, 
                       help="MLflow run name")
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(training_config.random_seed)
    
    # Update configurations if provided
    if args.epochs:
        model_config.num_epochs = args.epochs
    if args.batch_size:
        model_config.batch_size = args.batch_size
    if args.lr:
        model_config.learning_rate = args.lr
    if args.device:
        model_config.device = args.device
    if args.experiment_name:
        training_config.experiment_name = args.experiment_name
    if args.run_name:
        training_config.run_name = args.run_name
    
    print("=" * 60)
    print("IMAGE-TEXT RANKING MODEL TRAINING")
    print("=" * 60)
    print(f"Device: {model_config.device}")
    print(f"Epochs: {model_config.num_epochs}")
    print(f"Batch size: {model_config.batch_size}")
    print(f"Learning rate: {model_config.learning_rate}")
    print(f"Embedding dimension: {model_config.embedding_dim}")
    print(f"Temperature: {model_config.temperature}")
    print("=" * 60)
    
    try:
        # Get data loaders
        print("Loading datasets...")
        dataloaders = get_dataloaders(args.data_dir)
        
        print(f"Train samples: {len(dataloaders['train'].dataset)}")
        print(f"Validation samples: {len(dataloaders['val'].dataset)}")
        print(f"Test samples: {len(dataloaders['test'].dataset)}")
        
        # Initialize trainer
        trainer = Trainer(dataloaders=dataloaders)
        
        # Resume from checkpoint if provided
        if args.resume:
            print(f"Resuming from checkpoint: {args.resume}")
            trainer.load_checkpoint(args.resume)
        
        # Start training
        print("\nStarting training...")
        training_history = trainer.train()
        
        print("\nTraining completed successfully!")
        print(f"Best validation Recall@10: {trainer.best_score:.4f}")
        print(f"Final model saved to: {data_config.models_path / 'final_model.pt'}")
        print(f"Training history saved to: {data_config.models_path / 'training_history.json'}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {e}")
        raise


if __name__ == "__main__":
    main() 