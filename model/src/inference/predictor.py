"""
Inference module for image-text ranking model.
"""

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
from typing import List, Dict, Tuple, Union, Optional
import json

from ..models import DualEncoder
from ..data import TextProcessor
from ..training import load_trained_model
from ..config import model_config, data_config


class ImageTextRankingPredictor:
    """Predictor class for image-text ranking."""
    
    def __init__(
        self,
        model_path: str = None,
        vocab_path: str = None,
        device: str = None
    ):
        self.device = device or model_config.device
        
        # Load model
        if model_path is None:
            model_path = str(data_config.models_path / "final_model.pt")
        
        self.model = load_trained_model(model_path, self.device)
        
        # Load text processor
        if vocab_path is None:
            vocab_path = str(data_config.processed_data_path / "vocabulary.json")
        
        self.text_processor = self._load_text_processor(vocab_path)
        
        # Image transforms
        self.image_transform = transforms.Compose([
            transforms.Resize(model_config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=model_config.image_mean, std=model_config.image_std)
        ])
        
        # Image database for ranking
        self.image_database = None
        self.image_embeddings = None
        
    def _load_text_processor(self, vocab_path: str) -> TextProcessor:
        """Load text processor with vocabulary."""
        text_processor = TextProcessor(
            vocab_size=model_config.vocab_size,
            max_length=model_config.max_text_length
        )
        
        with open(vocab_path, 'r') as f:
            vocab_data = json.load(f)
        
        text_processor.word_to_idx = {k: int(v) for k, v in vocab_data['word_to_idx'].items()}
        text_processor.idx_to_word = {int(k): v for k, v in vocab_data['idx_to_word'].items()}
        text_processor.vocab_built = True
        
        return text_processor
    
    def encode_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Encode text query to embedding.
        
        Args:
            text: Single text string or list of texts
            
        Returns:
            Text embeddings [N, D]
        """
        if isinstance(text, str):
            text = [text]
        
        # Tokenize texts
        tokenized = []
        for t in text:
            tokens = self.text_processor.encode(t)
            tokenized.append(tokens)
        
        # Convert to tensor
        text_tokens = torch.tensor(tokenized, dtype=torch.long).to(self.device)
        
        # Encode
        with torch.no_grad():
            text_embeddings = self.model.encode_text(text_tokens)
        
        return text_embeddings.cpu().numpy()
    
    def encode_image(self, images: Union[str, Image.Image, List[Union[str, Image.Image]]]) -> np.ndarray:
        """
        Encode image(s) to embedding.
        
        Args:
            images: Single image path/PIL Image or list of images
            
        Returns:
            Image embeddings [N, D]
        """
        if not isinstance(images, list):
            images = [images]
        
        # Load and preprocess images
        processed_images = []
        for img in images:
            if isinstance(img, str):
                img = Image.open(img).convert('RGB')
            elif isinstance(img, Path):
                img = Image.open(str(img)).convert('RGB')
            
            processed_img = self.image_transform(img)
            processed_images.append(processed_img)
        
        # Stack images
        image_batch = torch.stack(processed_images).to(self.device)
        
        # Encode
        with torch.no_grad():
            image_embeddings = self.model.encode_image(image_batch)
        
        return image_embeddings.cpu().numpy()
    
    def compute_similarity(
        self, 
        text_embeddings: np.ndarray, 
        image_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute similarity between text and image embeddings.
        
        Args:
            text_embeddings: [N_text, D]
            image_embeddings: [N_images, D]
            
        Returns:
            Similarity matrix [N_text, N_images]
        """
        # Cosine similarity
        text_norm = np.linalg.norm(text_embeddings, axis=1, keepdims=True)
        image_norm = np.linalg.norm(image_embeddings, axis=1, keepdims=True)
        
        normalized_text = text_embeddings / text_norm
        normalized_image = image_embeddings / image_norm
        
        similarity = np.dot(normalized_text, normalized_image.T)
        
        return similarity
    
    def rank_images(
        self,
        text_query: str,
        image_paths: List[str],
        top_k: int = 10
    ) -> List[Dict]:
        """
        Rank images based on text query.
        
        Args:
            text_query: Text query string
            image_paths: List of image file paths
            top_k: Number of top results to return
            
        Returns:
            List of ranking results
        """
        # Encode text query
        text_embeddings = self.encode_text(text_query)
        
        # Encode images
        image_embeddings = self.encode_image(image_paths)
        
        # Compute similarities
        similarities = self.compute_similarity(text_embeddings, image_embeddings)
        similarities = similarities[0]  # Single query
        
        # Get top-k indices
        top_indices = np.argsort(-similarities)[:top_k]
        
        # Format results
        results = []
        for rank, idx in enumerate(top_indices):
            results.append({
                'rank': rank + 1,
                'image_path': image_paths[idx],
                'similarity': float(similarities[idx]),
                'score': float(similarities[idx])
            })
        
        return results
    
    def rank_texts(
        self,
        image_path: str,
        text_queries: List[str],
        top_k: int = 10
    ) -> List[Dict]:
        """
        Rank texts based on image query.
        
        Args:
            image_path: Path to image file
            text_queries: List of text queries
            top_k: Number of top results to return
            
        Returns:
            List of ranking results
        """
        # Encode image
        image_embeddings = self.encode_image(image_path)
        
        # Encode texts
        text_embeddings = self.encode_text(text_queries)
        
        # Compute similarities
        similarities = self.compute_similarity(text_embeddings, image_embeddings)
        similarities = similarities[:, 0]  # Single image
        
        # Get top-k indices
        top_indices = np.argsort(-similarities)[:top_k]
        
        # Format results
        results = []
        for rank, idx in enumerate(top_indices):
            results.append({
                'rank': rank + 1,
                'text': text_queries[idx],
                'similarity': float(similarities[idx]),
                'score': float(similarities[idx])
            })
        
        return results
    
    def build_image_database(
        self,
        image_paths: List[str],
        batch_size: int = 32
    ) -> None:
        """
        Pre-compute embeddings for a database of images.
        
        Args:
            image_paths: List of image file paths
            batch_size: Batch size for encoding
        """
        print(f"Building image database with {len(image_paths)} images...")
        
        self.image_database = image_paths
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_embeddings = self.encode_image(batch_paths)
            all_embeddings.append(batch_embeddings)
        
        # Concatenate all embeddings
        self.image_embeddings = np.concatenate(all_embeddings, axis=0)
        
        print("Image database built successfully!")
    
    def search_database(
        self,
        text_query: str,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Search pre-built image database with text query.
        
        Args:
            text_query: Text query string
            top_k: Number of top results to return
            
        Returns:
            List of search results
        """
        if self.image_database is None or self.image_embeddings is None:
            raise ValueError("Image database not built. Call build_image_database first.")
        
        # Encode text query
        text_embeddings = self.encode_text(text_query)
        
        # Compute similarities with database
        similarities = self.compute_similarity(text_embeddings, self.image_embeddings)
        similarities = similarities[0]  # Single query
        
        # Get top-k indices
        top_indices = np.argsort(-similarities)[:top_k]
        
        # Format results
        results = []
        for rank, idx in enumerate(top_indices):
            results.append({
                'rank': rank + 1,
                'image_path': self.image_database[idx],
                'similarity': float(similarities[idx]),
                'score': float(similarities[idx])
            })
        
        return results
    
    def save_database(self, save_path: str) -> None:
        """Save pre-computed image database."""
        if self.image_database is None or self.image_embeddings is None:
            raise ValueError("No database to save")
        
        np.savez(
            save_path,
            image_paths=self.image_database,
            embeddings=self.image_embeddings
        )
        print(f"Database saved to {save_path}")
    
    def load_database(self, load_path: str) -> None:
        """Load pre-computed image database."""
        data = np.load(load_path, allow_pickle=True)
        self.image_database = data['image_paths'].tolist()
        self.image_embeddings = data['embeddings']
        print(f"Database loaded from {load_path}")


def create_predictor(
    model_path: str = None,
    vocab_path: str = None,
    device: str = None
) -> ImageTextRankingPredictor:
    """Create a predictor instance."""
    return ImageTextRankingPredictor(
        model_path=model_path,
        vocab_path=vocab_path,
        device=device
    ) 