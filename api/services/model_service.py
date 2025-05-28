"""
Model service for handling inference operations.
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from src.config import data_config, model_config
from src.inference.predictor import create_predictor, ImageTextRankingPredictor

from api_models.schemas import ModelInfo, RankingResult

logger = logging.getLogger(__name__)


class ModelService:
    """Service for managing model inference operations."""
    
    def __init__(self):
        self.predictor: Optional[ImageTextRankingPredictor] = None
        self.model_info: Dict[str, Any] = {}
        self.loaded_at: Optional[datetime] = None
        
    async def load_model(self, model_path: str = None, vocab_path: str = None) -> bool:
        """
        Load the trained model.
        
        Args:
            model_path: Path to model file
            vocab_path: Path to vocabulary file
            
        Returns:
            True if model loaded successfully
        """
        try:
            logger.info("Loading image-text ranking model...")
            
            # Use default paths if not provided
            if model_path is None:
                model_path = str(data_config.models_path / "final_model.pt")
            
            if vocab_path is None:
                vocab_path = str(data_config.processed_data_path / "vocabulary.json")
            
            # Check if files exist
            if not Path(model_path).exists():
                logger.warning(f"Model file not found: {model_path}")
                # Try alternative paths
                alt_paths = [
                    "/model/models/final_model.pt",
                    "./models/final_model.pt"
                ]
                logger.error(f"Trying alternative paths: {alt_paths}")
                
                for alt_path in alt_paths:
                    if Path(alt_path).exists():
                        model_path = alt_path
                        logger.error(f"Found model at alternative path: {model_path}")
                        break
                else:
                    logger.error("No trained model found. Please train a model first.")
                    return False
            
            if not Path(vocab_path).exists():
                logger.warning(f"Vocabulary file not found: {vocab_path}")
                # Try alternative paths
                alt_vocab_paths = [
                    "/app/model/data/processed/vocabulary.json",
                    "/app/data/processed/vocabulary.json",
                    "./data/processed/vocabulary.json"
                ]
                
                for alt_path in alt_vocab_paths:
                    if Path(alt_path).exists():
                        vocab_path = alt_path
                        logger.info(f"Found vocabulary at alternative path: {vocab_path}")
                        break
                else:
                    logger.error("No vocabulary file found. Please process data first.")
                    return False
            
            # Create predictor
            self.predictor = create_predictor(
                model_path=model_path,
                vocab_path=vocab_path,
                device=model_config.device
            )
            
            # Store model info
            model_file = Path(model_path)
            vocab_file = Path(vocab_path)
            
            self.model_info = {
                "model_name": "ImageTextRankingModel",
                "model_path": str(model_file.absolute()),
                "vocab_path": str(vocab_file.absolute()),
                "device": model_config.device,
                "embedding_dim": model_config.embedding_dim,
                "vocab_size": model_config.vocab_size,
                "model_size_mb": round(model_file.stat().st_size / (1024 * 1024), 2),
                "loaded_at": datetime.now()
            }
            
            self.loaded_at = datetime.now()
            
            logger.info(f"✅ Model loaded successfully on device: {model_config.device}")
            logger.info(f"Model size: {self.model_info['model_size_mb']} MB")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.predictor = None
            self.model_info = {}
            return False
    
    async def reload_model(self) -> bool:
        """Reload the current model."""
        if not self.model_info:
            return await self.load_model()
        
        model_path = self.model_info.get("model_path")
        vocab_path = self.model_info.get("vocab_path")
        
        return await self.load_model(model_path, vocab_path)
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.predictor is not None
    
    def get_model_info(self) -> ModelInfo:
        """Get model information."""
        if not self.is_model_loaded():
            raise RuntimeError("No model loaded")
        
        return ModelInfo(**self.model_info)
    
    def get_device(self) -> str:
        """Get the device being used."""
        return model_config.device if self.is_model_loaded() else "none"
    
    async def rank_images(
        self,
        text_query: str,
        image_paths: List[str],
        top_k: int = 10
    ) -> List[RankingResult]:
        """
        Rank images based on text query.
        
        Args:
            text_query: Text query string
            image_paths: List of image file paths
            top_k: Number of top results to return
            
        Returns:
            List of ranking results
        """
        if not self.is_model_loaded():
            raise RuntimeError("Model not loaded")
        
        start_time = time.time()
        
        try:
            # Filter existing image paths
            existing_paths = [path for path in image_paths if Path(path).exists()]
            
            if not existing_paths:
                logger.warning("No valid image paths found")
                return []
            
            if len(existing_paths) < len(image_paths):
                logger.warning(f"Only {len(existing_paths)}/{len(image_paths)} image paths exist")
            
            # Perform ranking
            results = self.predictor.rank_images(
                text_query=text_query,
                image_paths=existing_paths,
                top_k=min(top_k, len(existing_paths))
            )
            
            # Convert to schema format
            ranking_results = [
                RankingResult(
                    rank=result["rank"],
                    score=result["score"],
                    image_path=result["image_path"]
                )
                for result in results
            ]
            
            elapsed_time = time.time() - start_time
            logger.info(f"Text-to-image ranking completed in {elapsed_time:.3f}s")
            
            return ranking_results
            
        except Exception as e:
            logger.error(f"Text-to-image ranking failed: {e}")
            raise
    
    async def rank_texts(
        self,
        image_path: str,
        texts: List[str],
        top_k: int = 10
    ) -> List[RankingResult]:
        """
        Rank texts based on image query.
        
        Args:
            image_path: Path to query image
            texts: List of text candidates
            top_k: Number of top results to return
            
        Returns:
            List of ranking results
        """
        if not self.is_model_loaded():
            raise RuntimeError("Model not loaded")
        
        start_time = time.time()
        
        try:
            # Check if image exists
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            # Perform ranking
            results = self.predictor.rank_texts(
                image_path=image_path,
                texts=texts,
                top_k=min(top_k, len(texts))
            )
            
            # Convert to schema format
            ranking_results = [
                RankingResult(
                    rank=result["rank"],
                    score=result["score"],
                    text=result["text"]
                )
                for result in results
            ]
            
            elapsed_time = time.time() - start_time
            logger.info(f"Image-to-text ranking completed in {elapsed_time:.3f}s")
            
            return ranking_results
            
        except Exception as e:
            logger.error(f"Image-to-text ranking failed: {e}")
            raise
    
    async def batch_ranking(
        self,
        text_queries: List[str],
        image_paths: List[str],
        top_k: int = 10
    ) -> List[List[RankingResult]]:
        """
        Perform batch ranking for multiple queries.
        
        Args:
            text_queries: List of text queries
            image_paths: List of image file paths
            top_k: Number of top results per query
            
        Returns:
            List of ranking results for each query
        """
        if not self.is_model_loaded():
            raise RuntimeError("Model not loaded")
        
        start_time = time.time()
        
        try:
            # Filter existing image paths
            existing_paths = [path for path in image_paths if Path(path).exists()]
            
            if not existing_paths:
                raise FileNotFoundError("No valid image paths found")
            
            # Process each query
            all_results = []
            
            for query in text_queries:
                query_results = await self.rank_images(
                    text_query=query,
                    image_paths=existing_paths,
                    top_k=top_k
                )
                all_results.append(query_results)
            
            elapsed_time = time.time() - start_time
            logger.info(f"Batch ranking completed in {elapsed_time:.3f}s for {len(text_queries)} queries")
            
            return all_results
            
        except Exception as e:
            logger.error(f"Batch ranking failed: {e}")
            raise
    
    async def find_similar_images(
        self,
        query_image_path: str,
        candidate_image_paths: List[str],
        top_k: int = 10
    ) -> List[RankingResult]:
        """
        Find similar images to a query image.
        
        Args:
            query_image_path: Path to query image
            candidate_image_paths: List of candidate image paths
            top_k: Number of top results to return
            
        Returns:
            List of similar images
        """
        if not self.is_model_loaded():
            raise RuntimeError("Model not loaded")
        
        start_time = time.time()
        
        try:
            # Check if query image exists
            if not Path(query_image_path).exists():
                raise FileNotFoundError(f"Query image not found: {query_image_path}")
            
            # Filter existing candidate paths
            existing_paths = [path for path in candidate_image_paths if Path(path).exists()]
            
            if not existing_paths:
                raise FileNotFoundError("No valid candidate image paths found")
            
            # Encode query image
            query_embedding = self.predictor.encode_image([query_image_path])
            
            # Encode candidate images
            candidate_embeddings = self.predictor.encode_image(existing_paths)
            
            # Compute similarities (excluding self if present)
            similarities = self.predictor.compute_similarity(query_embedding, candidate_embeddings)
            similarities = similarities[0]  # Get similarities for the single query
            
            # Create results with rankings
            image_similarities = [
                {
                    "image_path": path,
                    "score": float(sim),
                    "is_self": path == query_image_path
                }
                for path, sim in zip(existing_paths, similarities)
            ]
            
            # Filter out self-matches and sort by similarity
            filtered_similarities = [
                item for item in image_similarities if not item["is_self"]
            ]
            
            sorted_similarities = sorted(
                filtered_similarities,
                key=lambda x: x["score"],
                reverse=True
            )
            
            # Take top k and format results
            top_results = sorted_similarities[:top_k]
            
            ranking_results = [
                RankingResult(
                    rank=idx + 1,
                    score=result["score"],
                    image_path=result["image_path"]
                )
                for idx, result in enumerate(top_results)
            ]
            
            elapsed_time = time.time() - start_time
            logger.info(f"Similar image search completed in {elapsed_time:.3f}s")
            
            return ranking_results
            
        except Exception as e:
            logger.error(f"Similar image search failed: {e}")
            raise
    
    async def encode_text(self, texts: List[str]) -> np.ndarray:
        """Encode text to embeddings."""
        if not self.is_model_loaded():
            raise RuntimeError("Model not loaded")
        
        return self.predictor.encode_text(texts)
    
    async def encode_image(self, image_paths: List[str]) -> np.ndarray:
        """Encode images to embeddings."""
        if not self.is_model_loaded():
            raise RuntimeError("Model not loaded")
        
        # Filter existing paths
        existing_paths = [path for path in image_paths if Path(path).exists()]
        
        if not existing_paths:
            raise FileNotFoundError("No valid image paths found")
        
        return self.predictor.encode_image(existing_paths)
    
    async def compute_similarity(
        self,
        text_embeddings: np.ndarray,
        image_embeddings: np.ndarray
    ) -> np.ndarray:
        """Compute similarity between text and image embeddings."""
        if not self.is_model_loaded():
            raise RuntimeError("Model not loaded")
        
        return self.predictor.compute_similarity(text_embeddings, image_embeddings) 