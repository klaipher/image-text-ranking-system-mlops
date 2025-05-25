"""
Encoder modules for the dual-encoder architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Tuple

from ..config import model_config


class ImageEncoder(nn.Module):
    """Image encoder using MobileNetV3 Small."""
    
    def __init__(self, embedding_dim: int = 256, pretrained: bool = True):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Load MobileNetV3 Small
        if pretrained:
            weights = models.MobileNet_V3_Small_Weights.DEFAULT
        else:
            weights = None
        self.backbone = models.mobilenet_v3_small(weights=weights)
        
        # Remove the classifier head
        self.backbone.classifier = nn.Identity()
        
        # Get the feature dimension from the last layer
        # MobileNetV3 Small has 576 features before classifier
        backbone_dim = 576
        
        # Add projection head
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, 512),
            nn.ReLU(),
            nn.Dropout(model_config.dropout_rate),
            nn.Linear(512, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of image encoder.
        
        Args:
            images: Batch of images [batch_size, 3, H, W]
            
        Returns:
            Image embeddings [batch_size, embedding_dim]
        """
        # Extract features
        features = self.backbone(images)  # [batch_size, 576]
        
        # Project to embedding space
        embeddings = self.projection(features)  # [batch_size, embedding_dim]
        
        # L2 normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings


class TextEncoder(nn.Module):
    """Simple text encoder with embedding and LSTM."""
    
    def __init__(
        self, 
        vocab_size: int = 10000,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        max_length: int = 64
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_length = max_length
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        
        # LSTM for sequence encoding
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=model_config.dropout_rate if num_layers > 1 else 0
        )
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(model_config.dropout_rate),
            nn.Linear(512, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
    def forward(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of text encoder.
        
        Args:
            text_tokens: Batch of tokenized text [batch_size, max_length]
            
        Returns:
            Text embeddings [batch_size, embedding_dim]
        """
        batch_size, seq_len = text_tokens.size()
        
        # Embedding
        embedded = self.embedding(text_tokens)  # [batch_size, seq_len, hidden_dim]
        
        # LSTM encoding
        lstm_out, (hidden, cell) = self.lstm(embedded)  # lstm_out: [batch_size, seq_len, hidden_dim]
        
        # Use mean pooling over sequence dimension (ignoring padding)
        # Create mask for padding tokens
        mask = (text_tokens != 0).float().unsqueeze(-1)  # [batch_size, seq_len, 1]
        
        # Apply mask and compute mean
        masked_output = lstm_out * mask
        sequence_lengths = mask.sum(dim=1)  # [batch_size, 1]
        mean_pooled = masked_output.sum(dim=1) / sequence_lengths.clamp(min=1)  # [batch_size, hidden_dim]
        
        # Project to embedding space
        embeddings = self.projection(mean_pooled)  # [batch_size, embedding_dim]
        
        # L2 normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings


class DualEncoder(nn.Module):
    """Dual encoder model for image-text ranking."""
    
    def __init__(
        self,
        vocab_size: int = None,
        embedding_dim: int = None,
        text_hidden_dim: int = None,
        temperature: float = None
    ):
        super().__init__()
        
        # Use config defaults if not provided
        vocab_size = vocab_size or model_config.vocab_size
        embedding_dim = embedding_dim or model_config.embedding_dim
        text_hidden_dim = text_hidden_dim or model_config.text_encoder_dim
        self.temperature = temperature or model_config.temperature
        
        # Initialize encoders
        self.image_encoder = ImageEncoder(embedding_dim=embedding_dim)
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=text_hidden_dim,
            max_length=model_config.max_text_length
        )
        
    def forward(
        self, 
        images: torch.Tensor, 
        text_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of dual encoder.
        
        Args:
            images: Batch of images [batch_size, 3, H, W]
            text_tokens: Batch of tokenized text [batch_size, max_length]
            
        Returns:
            Tuple of (image_embeddings, text_embeddings)
        """
        image_embeddings = self.image_encoder(images)
        text_embeddings = self.text_encoder(text_tokens)
        
        return image_embeddings, text_embeddings
    
    def compute_similarity(
        self, 
        image_embeddings: torch.Tensor, 
        text_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute similarity matrix between image and text embeddings.
        
        Args:
            image_embeddings: [batch_size, embedding_dim]
            text_embeddings: [batch_size, embedding_dim]
            
        Returns:
            Similarity matrix [batch_size, batch_size]
        """
        # Compute cosine similarity
        similarity_matrix = torch.matmul(image_embeddings, text_embeddings.t())
        
        # Apply temperature scaling
        similarity_matrix = similarity_matrix / self.temperature
        
        return similarity_matrix
    
    def contrastive_loss(
        self, 
        image_embeddings: torch.Tensor, 
        text_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss for image-text pairs.
        
        Args:
            image_embeddings: [batch_size, embedding_dim]
            text_embeddings: [batch_size, embedding_dim]
            
        Returns:
            Contrastive loss scalar
        """
        batch_size = image_embeddings.size(0)
        
        # Compute similarity matrix
        similarity_matrix = self.compute_similarity(image_embeddings, text_embeddings)
        
        # Create labels (diagonal should be positive pairs)
        labels = torch.arange(batch_size, device=image_embeddings.device)
        
        # Compute cross-entropy loss for both directions
        loss_i2t = F.cross_entropy(similarity_matrix, labels)
        loss_t2i = F.cross_entropy(similarity_matrix.t(), labels)
        
        # Average the losses
        total_loss = (loss_i2t + loss_t2i) / 2
        
        return total_loss
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to embeddings."""
        return self.image_encoder(images)
    
    def encode_text(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """Encode text to embeddings."""
        return self.text_encoder(text_tokens)


def create_model(config: dict = None) -> DualEncoder:
    """Create a dual encoder model with given configuration."""
    if config is None:
        config = {}
    
    model = DualEncoder(
        vocab_size=config.get('vocab_size', model_config.vocab_size),
        embedding_dim=config.get('embedding_dim', model_config.embedding_dim),
        text_hidden_dim=config.get('text_hidden_dim', model_config.text_encoder_dim),
        temperature=config.get('temperature', model_config.temperature)
    )
    
    return model 