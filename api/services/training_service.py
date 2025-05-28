"""
Training service for handling model training operations.
"""

import os
import sys
import uuid
import asyncio
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
import json

# Add model src to path
sys.path.append('/app/model/src')

from models.schemas import TrainingStatus, TrainingRun, TrainingHistory

logger = logging.getLogger(__name__)


class TrainingService:
    """Service for managing model training operations."""
    
    def __init__(self):
        self.current_training: Optional[Dict[str, Any]] = None
        self.training_history: List[TrainingRun] = []
        self.training_thread: Optional[threading.Thread] = None
        self.training_process: Optional[subprocess.Popen] = None
        
        # Load existing training history
        self._load_training_history()
    
    def _load_training_history(self):
        """Load training history from disk."""
        try:
            history_file = Path("/app/training_history.json")
            if history_file.exists():
                with open(history_file, 'r') as f:
                    history_data = json.load(f)
                
                self.training_history = [
                    TrainingRun(**run_data) for run_data in history_data
                ]
                
                logger.info(f"Loaded {len(self.training_history)} training runs from history")
            
        except Exception as e:
            logger.warning(f"Failed to load training history: {e}")
            self.training_history = []
    
    def _save_training_history(self):
        """Save training history to disk."""
        try:
            history_file = Path("/app/training_history.json")
            history_data = [run.dict() for run in self.training_history]
            
            with open(history_file, 'w') as f:
                json.dump(history_data, f, indent=2, default=str)
                
            logger.info("Training history saved")
            
        except Exception as e:
            logger.error(f"Failed to save training history: {e}")
    
    def is_training_active(self) -> bool:
        """Check if training is currently active."""
        return self.current_training is not None and self.current_training.get("status") == "running"
    
    def get_training_status(self) -> TrainingStatus:
        """Get current training status."""
        if not self.current_training:
            return TrainingStatus(
                is_active=False,
                status="not_started"
            )
        
        return TrainingStatus(
            is_active=self.is_training_active(),
            training_id=self.current_training.get("training_id"),
            status=self.current_training.get("status"),
            current_epoch=self.current_training.get("current_epoch"),
            total_epochs=self.current_training.get("total_epochs"),
            current_loss=self.current_training.get("current_loss"),
            best_score=self.current_training.get("best_score"),
            elapsed_time=self.current_training.get("elapsed_time"),
            estimated_remaining=self.current_training.get("estimated_remaining"),
            last_update=self.current_training.get("last_update")
        )
    
    def get_training_history(self) -> TrainingHistory:
        """Get training history."""
        # Find last successful run
        last_successful = None
        for run in reversed(self.training_history):
            if run.status == "completed":
                last_successful = run
                break
        
        return TrainingHistory(
            total_runs=len(self.training_history),
            runs=self.training_history,
            last_successful_run=last_successful
        )
    
    async def start_training(
        self,
        config: Dict[str, Any],
        background_tasks=None
    ) -> str:
        """
        Start model training.
        
        Args:
            config: Training configuration
            background_tasks: FastAPI background tasks
            
        Returns:
            Training job ID
        """
        if self.is_training_active():
            raise RuntimeError("Training already in progress")
        
        # Generate training ID
        training_id = str(uuid.uuid4())
        
        # Create training run record
        training_run = TrainingRun(
            run_id=training_id,
            experiment_name=config.get("experiment_name", "api_training"),
            run_name=config.get("run_name", f"api_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            status="starting",
            start_time=datetime.now(),
            config=config
        )
        
        # Initialize current training state
        self.current_training = {
            "training_id": training_id,
            "status": "starting",
            "start_time": datetime.now(),
            "config": config,
            "current_epoch": 0,
            "total_epochs": config.get("epochs", 10),
            "current_loss": None,
            "best_score": None,
            "elapsed_time": 0,
            "estimated_remaining": None,
            "last_update": datetime.now()
        }
        
        # Add to history
        self.training_history.append(training_run)
        self._save_training_history()
        
        # Start training in background
        if background_tasks:
            background_tasks.add_task(self._run_training, training_id, config)
        else:
            # Start in separate thread
            self.training_thread = threading.Thread(
                target=asyncio.run,
                args=(self._run_training(training_id, config),)
            )
            self.training_thread.start()
        
        logger.info(f"Training started with ID: {training_id}")
        
        return training_id
    
    async def _run_training(self, training_id: str, config: Dict[str, Any]):
        """Run the actual training process."""
        try:
            logger.info(f"Starting training process for ID: {training_id}")
            
            # Update status
            self.current_training["status"] = "running"
            self.current_training["last_update"] = datetime.now()
            
            # Prepare training command
            train_cmd = self._prepare_training_command(config)
            
            logger.info(f"Training command: {' '.join(train_cmd)}")
            
            # Start training process
            self.training_process = subprocess.Popen(
                train_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                cwd="/app/model"
            )
            
            # Monitor training process
            await self._monitor_training_process()
            
            # Check final result
            if self.training_process.returncode == 0:
                self._update_training_status("completed")
                logger.info(f"Training {training_id} completed successfully")
            else:
                self._update_training_status("failed")
                logger.error(f"Training {training_id} failed with return code {self.training_process.returncode}")
            
        except Exception as e:
            self._update_training_status("failed", error=str(e))
            logger.error(f"Training {training_id} failed with exception: {e}")
        
        finally:
            # Clean up
            self.training_process = None
            
            # Update history
            self._update_training_history()
    
    def _prepare_training_command(self, config: Dict[str, Any]) -> list:
        """Prepare the training command."""
        cmd = ["python", "train.py"]
        
        # Add configuration parameters
        if config.get("epochs"):
            cmd.extend(["--epochs", str(config["epochs"])])
        
        if config.get("batch_size"):
            cmd.extend(["--batch-size", str(config["batch_size"])])
        
        if config.get("learning_rate"):
            cmd.extend(["--lr", str(config["learning_rate"])])
        
        if config.get("experiment_name"):
            cmd.extend(["--experiment-name", config["experiment_name"]])
        
        if config.get("run_name"):
            cmd.extend(["--run-name", config["run_name"]])
        
        # Add data directory if specified
        data_source = config.get("data_source", "local")
        if data_source == "local":
            cmd.extend(["--data-dir", "/app/model/data"])
        elif data_source == "minio":
            # Download data from MinIO first if needed
            cmd.extend(["--data-dir", "/app/data/processed"])
        
        return cmd
    
    async def _monitor_training_process(self):
        """Monitor the training process and update status."""
        start_time = time.time()
        
        while self.training_process and self.training_process.poll() is None:
            # Update elapsed time
            elapsed = time.time() - start_time
            self.current_training["elapsed_time"] = elapsed
            self.current_training["last_update"] = datetime.now()
            
            # Read output for progress information
            try:
                # Non-blocking read
                output = self.training_process.stdout.readline()
                if output:
                    self._parse_training_output(output.strip())
                
            except Exception as e:
                logger.warning(f"Error reading training output: {e}")
            
            # Sleep briefly
            await asyncio.sleep(1)
        
        # Final status update
        if self.training_process:
            elapsed = time.time() - start_time
            self.current_training["elapsed_time"] = elapsed
            self.current_training["last_update"] = datetime.now()
    
    def _parse_training_output(self, output: str):
        """Parse training output for progress information."""
        try:
            # Look for epoch information
            if "Epoch" in output and "/" in output:
                # Example: "Epoch 3/10"
                parts = output.split()
                for part in parts:
                    if "/" in part and part.replace("/", "").replace("-", "").isdigit():
                        current, total = part.split("/")
                        self.current_training["current_epoch"] = int(current)
                        self.current_training["total_epochs"] = int(total)
                        break
            
            # Look for loss information
            if "loss:" in output.lower():
                # Example: "Train Loss: 0.245"
                try:
                    loss_idx = output.lower().find("loss:")
                    loss_part = output[loss_idx:].split()[1]
                    loss_val = float(loss_part.replace(",", ""))
                    self.current_training["current_loss"] = loss_val
                except:
                    pass
            
            # Look for validation score
            if "recall@10" in output.lower() or "best score" in output.lower():
                try:
                    # Extract numerical value
                    import re
                    numbers = re.findall(r'\d+\.?\d*', output)
                    if numbers:
                        score = float(numbers[-1])
                        if score <= 1.0:  # Assume it's a ratio
                            self.current_training["best_score"] = score
                except:
                    pass
                    
        except Exception as e:
            logger.debug(f"Error parsing training output: {e}")
    
    def _update_training_status(self, status: str, error: str = None):
        """Update training status."""
        if self.current_training:
            self.current_training["status"] = status
            self.current_training["last_update"] = datetime.now()
            
            if error:
                self.current_training["error"] = error
    
    def _update_training_history(self):
        """Update the training history with final results."""
        if not self.current_training:
            return
        
        training_id = self.current_training["training_id"]
        
        # Find the corresponding run in history
        for run in self.training_history:
            if run.run_id == training_id:
                # Update with final results
                run.status = self.current_training["status"]
                run.end_time = datetime.now()
                run.duration = self.current_training.get("elapsed_time")
                run.final_loss = self.current_training.get("current_loss")
                run.best_score = self.current_training.get("best_score")
                break
        
        # Save updated history
        self._save_training_history()
    
    async def stop_training(self):
        """Stop current training."""
        if not self.is_training_active():
            raise RuntimeError("No active training to stop")
        
        logger.info("Stopping training...")
        
        # Terminate training process
        if self.training_process:
            self.training_process.terminate()
            
            # Wait a bit for graceful termination
            try:
                self.training_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't terminate gracefully
                self.training_process.kill()
                self.training_process.wait()
        
        # Update status
        self._update_training_status("stopped")
        self._update_training_history()
        
        # Clear current training
        self.current_training = None
        
        logger.info("Training stopped")
    
    def _cleanup_old_history(self, max_runs: int = 50):
        """Clean up old training history to prevent unlimited growth."""
        if len(self.training_history) > max_runs:
            # Keep the most recent runs
            self.training_history = self.training_history[-max_runs:]
            self._save_training_history()
            logger.info(f"Cleaned up training history, keeping {max_runs} most recent runs") 