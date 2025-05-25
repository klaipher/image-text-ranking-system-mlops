"""
Training module for the image-text ranking model.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import mlflow
import mlflow.pytorch
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import time
import json
import warnings

from ..models import DualEncoder, create_model
from ..data import get_dataloaders
from ..config import model_config, training_config, data_config
from .metrics import compute_retrieval_metrics


class Trainer:
    """Trainer class for the dual encoder model."""
    
    def __init__(
        self,
        model: Optional[DualEncoder] = None,
        dataloaders: Optional[Dict[str, DataLoader]] = None,
        device: str = None
    ):
        self.device = device or model_config.device
        
        # Initialize model
        self.model = model or create_model()
        self.model.to(self.device)
        
        # Initialize dataloaders
        self.dataloaders = dataloaders or get_dataloaders()
        
        # Initialize optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=model_config.learning_rate,
            weight_decay=model_config.weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=model_config.num_epochs
        )
        
        # Training state
        self.current_epoch = 0
        self.best_score = 0.0
        self.training_history = []
        
        # Create model save directory
        data_config.models_path.mkdir(parents=True, exist_ok=True)
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.dataloaders['train'], 
            desc=f"Epoch {self.current_epoch + 1}/{model_config.num_epochs}"
        )
        
        for batch_idx, (images, text_tokens, _, _) in enumerate(progress_bar):
            # Move to device
            images = images.to(self.device)
            text_tokens = text_tokens.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            image_embeddings, text_embeddings = self.model(images, text_tokens)
            
            # Compute loss
            loss = self.model.contrastive_loss(image_embeddings, text_embeddings)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss / num_batches:.4f}"
            })
            
            # Log every N steps
            if batch_idx % training_config.log_every_n_steps == 0:
                mlflow.log_metric("train_loss_step", loss.item(), 
                                step=self.current_epoch * len(self.dataloaders['train']) + batch_idx)
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {"train_loss": avg_loss}
    
    def evaluate(self, split: str = 'val') -> Dict[str, float]:
        """Evaluate the model on validation or test set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        # Collect all embeddings for retrieval evaluation
        all_image_embeddings = []
        all_text_embeddings = []
        all_image_files = []
        all_captions = []
        
        with torch.no_grad():
            for images, text_tokens, image_files, captions in tqdm(
                self.dataloaders[split], desc=f"Evaluating {split}"
            ):
                # Move to device
                images = images.to(self.device)
                text_tokens = text_tokens.to(self.device)
                
                # Forward pass
                image_embeddings, text_embeddings = self.model(images, text_tokens)
                
                # Compute loss
                loss = self.model.contrastive_loss(image_embeddings, text_embeddings)
                total_loss += loss.item()
                num_batches += 1
                
                # Collect embeddings
                all_image_embeddings.append(image_embeddings.cpu())
                all_text_embeddings.append(text_embeddings.cpu())
                all_image_files.extend(image_files)
                all_captions.extend(captions)
        
        # Concatenate all embeddings
        all_image_embeddings = torch.cat(all_image_embeddings, dim=0)
        all_text_embeddings = torch.cat(all_text_embeddings, dim=0)
        
        # Compute retrieval metrics
        retrieval_metrics = compute_retrieval_metrics(
            all_image_embeddings.numpy(),
            all_text_embeddings.numpy(),
            model_config.recall_at_k
        )
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        metrics = {
            f"{split}_loss": avg_loss,
            **{f"{split}_{k}": v for k, v in retrieval_metrics.items()}
        }
        
        return metrics
    
    def train(self) -> Dict[str, List[float]]:
        """Full training loop."""
        print(f"Starting training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Setup MLflow tracking
        mlflow.set_tracking_uri(training_config.tracking_uri)
        mlflow.set_experiment(training_config.experiment_name)
        
        with mlflow.start_run(run_name=training_config.run_name):
            # Log hyperparameters
            mlflow.log_params({
                "learning_rate": model_config.learning_rate,
                "batch_size": model_config.batch_size,
                "embedding_dim": model_config.embedding_dim,
                "num_epochs": model_config.num_epochs,
                "temperature": model_config.temperature
            })
            
            for epoch in range(model_config.num_epochs):
                self.current_epoch = epoch
                start_time = time.time()
                
                # Training
                train_metrics = self.train_epoch()
                
                # Validation
                if epoch % training_config.eval_every_n_epochs == 0:
                    val_metrics = self.evaluate('val')
                else:
                    val_metrics = {}
                
                # Update scheduler
                self.scheduler.step()
                
                # Combine metrics
                epoch_metrics = {**train_metrics, **val_metrics}
                epoch_metrics['epoch'] = epoch
                epoch_metrics['lr'] = self.optimizer.param_groups[0]['lr']
                epoch_metrics['epoch_time'] = time.time() - start_time
                
                self.training_history.append(epoch_metrics)
                
                # Log to MLflow
                for key, value in epoch_metrics.items():
                    if key != 'epoch':
                        mlflow.log_metric(key, value, step=epoch)
                
                # Print epoch summary
                print(f"\nEpoch {epoch + 1}/{model_config.num_epochs} Summary:")
                print(f"  Train Loss: {train_metrics['train_loss']:.4f}")
                if 'val_recall@10' in val_metrics:
                    print(f"  Val Recall@10: {val_metrics['val_recall@10']:.4f}")
                    print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
                print(f"  Learning Rate: {epoch_metrics['lr']:.6f}")
                print(f"  Epoch Time: {epoch_metrics['epoch_time']:.2f}s")
                
                # Save model checkpoints
                if training_config.save_every_n_epochs > 0 and (epoch + 1) % training_config.save_every_n_epochs == 0:
                    self._save_checkpoint(epoch)
                
                # Save best model
                if training_config.save_best_model and 'val_recall@10' in val_metrics:
                    current_score = val_metrics['val_recall@10']
                    if current_score > self.best_score:
                        self.best_score = current_score
                        self._save_best_model()
                        print(f"  New best model saved! Recall@10: {self.best_score:.4f}")
                
                # Early stopping
                if self._should_early_stop():
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    break
            
            # Final evaluation on test set
            print("\nEvaluating on test set...")
            test_metrics = self.evaluate('test')
            
            # Log final test metrics
            for key, value in test_metrics.items():
                mlflow.log_metric(f"final_{key}", value)
            
            print("\nFinal Test Results:")
            for key, value in test_metrics.items():
                print(f"  {key}: {value:.4f}")
            
            # Save final model
            self._save_final_model()
            
            # Log model to MLflow
            # Suppress pip version warning
            warnings.filterwarnings("ignore", message="Failed to resolve installed pip version.*")
            import torch
            # Example: (image_tensor, text_tensor)
            input_example = (
                torch.randn(1, 3, 224, 224),
                torch.randint(0, model_config.vocab_size, (1, model_config.max_text_length))
            )
            mlflow.pytorch.log_model(self.model, "model", input_example=input_example)
        
        return self.training_history
    
    def _save_checkpoint(self, epoch: int) -> None:
        """Save model checkpoint."""
        checkpoint_path = data_config.models_path / f"checkpoint_epoch_{epoch + 1}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_score': self.best_score,
            'training_history': self.training_history
        }, checkpoint_path)
    
    def _save_best_model(self) -> None:
        """Save the best model."""
        best_model_path = data_config.models_path / "final_model.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'best_score': self.best_score,
            'config': {
                'vocab_size': model_config.vocab_size,
                'embedding_dim': model_config.embedding_dim,
                'text_hidden_dim': model_config.text_encoder_dim,
                'temperature': model_config.temperature
            }
        }, best_model_path)
    
    def _save_final_model(self) -> None:
        """Save the final model."""
        final_model_path = data_config.models_path / "final_model.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'training_history': self.training_history,
            'config': {
                'vocab_size': model_config.vocab_size,
                'embedding_dim': model_config.embedding_dim,
                'text_hidden_dim': model_config.text_encoder_dim,
                'temperature': model_config.temperature
            }
        }, final_model_path)
        
        # Also save training history as JSON
        history_path = data_config.models_path / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
    
    def _should_early_stop(self) -> bool:
        """Check if early stopping should be triggered."""
        if training_config.early_stopping_patience <= 0:
            return False
        
        if len(self.training_history) < training_config.early_stopping_patience:
            return False
        
        recent_scores = []
        for i in range(training_config.early_stopping_patience):
            idx = -(i + 1)
            if f'val_recall@10' in self.training_history[idx]:
                recent_scores.append(self.training_history[idx]['val_recall@10'])
        
        if len(recent_scores) < training_config.early_stopping_patience:
            return False
        
        # Check if no improvement in recent epochs
        best_recent = max(recent_scores)
        return best_recent <= self.best_score
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_score = checkpoint.get('best_score', 0.0)
        self.training_history = checkpoint.get('training_history', [])
        
        print(f"Loaded checkpoint from epoch {self.current_epoch + 1}")


def load_trained_model(model_path: str, device: str = None) -> DualEncoder:
    """Load a trained model from file."""
    device = device or model_config.device
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    
    # Create model with saved config
    model = create_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model 