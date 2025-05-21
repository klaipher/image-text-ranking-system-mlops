import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from transformers import DistilBertModel, DistilBertConfig

class TextEncoder(nn.Module):
    def __init__(self, embedding_dim=512):
        super(TextEncoder, self).__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.projection = nn.Linear(768, embedding_dim)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use CLS token as sentence embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        projected = self.projection(cls_embedding)
        return F.normalize(projected, p=2, dim=1)

class ImageEncoder(nn.Module):
    def __init__(self, embedding_dim=512):
        super(ImageEncoder, self).__init__()
        # Load MobileNetV3-small with pre-trained weights
        mobilenet = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        # Remove the classifier layer
        self.backbone = nn.Sequential(*list(mobilenet.children())[:-1])
        # Get the output dimension of MobileNetV3-small
        self.in_features = mobilenet.classifier[0].in_features
        # Add a projection layer
        self.projection = nn.Linear(self.in_features, embedding_dim)
        
    def forward(self, images):
        features = self.backbone(images)
        # Flatten the features
        features = features.reshape(features.size(0), -1)
        # Project to the common embedding space
        projected = self.projection(features)
        return F.normalize(projected, p=2, dim=1)

class ImageTextRankingModel(nn.Module):
    def __init__(self, embedding_dim=512, temperature=0.07):
        super(ImageTextRankingModel, self).__init__()
        self.image_encoder = ImageEncoder(embedding_dim)
        self.text_encoder = TextEncoder(embedding_dim)
        self.temperature = temperature
        
    def forward(self, images, input_ids, attention_mask):
        # Encode images and text
        image_embeddings = self.image_encoder(images)
        text_embeddings = self.text_encoder(input_ids, attention_mask)
        
        # Calculate similarity matrix
        similarity = torch.matmul(text_embeddings, image_embeddings.t()) / self.temperature
        
        return similarity
    
    def get_image_embeddings(self, images):
        return self.image_encoder(images)
    
    def get_text_embeddings(self, input_ids, attention_mask):
        return self.text_encoder(input_ids, attention_mask) 