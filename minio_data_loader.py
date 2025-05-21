import os
import io
import json
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import DistilBertTokenizer
from minio import Minio
import pandas as pd

class MinIOFlickr8kDataset(Dataset):
    """
    Dataset class that loads Flickr8k data from MinIO storage
    """
    def __init__(self, split='train', transform=None, max_length=128):
        """
        Initialize the dataset
        
        Args:
            split (str): Dataset split ('train', 'val', or 'test')
            transform (callable, optional): Transform to apply to images
            max_length (int): Maximum token length for text
        """
        self.split = split
        self.transform = transform
        self.max_length = max_length
        
        # Default transforms if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        
        # Initialize tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        
        # Connect to MinIO
        self.minio_client = Minio(
            os.environ.get('MINIO_ENDPOINT', 'localhost:9000'),
            access_key=os.environ.get('MINIO_ACCESS_KEY', 'minioadmin'),
            secret_key=os.environ.get('MINIO_SECRET_KEY', 'minioadmin'),
            secure=False
        )
        
        self.bucket_name = os.environ.get('MINIO_BUCKET', 'flickr8k')
        
        # Load dataset splits
        self._load_dataset()
    
    def _load_dataset(self):
        """
        Load dataset from MinIO storage
        """
        try:
            # Get captions file
            captions_obj = self.minio_client.get_object(
                self.bucket_name,
                'text/Flickr8k.token.txt'
            )
            captions_content = captions_obj.read().decode('utf-8')
            
            # Parse captions
            self.captions = {}
            for line in captions_content.strip().split('\n'):
                img_id, caption = line.split('\t')
                img_id = img_id.split('#')[0]  # Remove caption number
                
                if img_id not in self.captions:
                    self.captions[img_id] = []
                self.captions[img_id].append(caption)
            
            # Get train/val/test splits
            splits_obj = None
            if self.split == 'train':
                splits_obj = self.minio_client.get_object(
                    self.bucket_name,
                    'text/Flickr_8k.trainImages.txt'
                )
            elif self.split == 'val':
                splits_obj = self.minio_client.get_object(
                    self.bucket_name,
                    'text/Flickr_8k.devImages.txt'
                )
            elif self.split == 'test':
                splits_obj = self.minio_client.get_object(
                    self.bucket_name,
                    'text/Flickr_8k.testImages.txt'
                )
            else:
                raise ValueError(f"Invalid split: {self.split}")
            
            splits_content = splits_obj.read().decode('utf-8')
            self.image_ids = [line.strip() for line in splits_content.split('\n') if line.strip()]
            
            # Prepare samples
            self.samples = []
            for img_id in self.image_ids:
                captions = self.captions.get(img_id, [])
                if captions:
                    self.samples.append({
                        'image_id': img_id,
                        'captions': captions
                    })
            
            print(f"Loaded {len(self.samples)} images for {self.split} split")
        
        except Exception as e:
            print(f"Error loading dataset from MinIO: {str(e)}")
            self.samples = []
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset
        
        Returns:
            tuple: (image_tensor, encoded_caption, image_id)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = self.samples[idx]
        image_id = sample['image_id']
        
        # Randomly select one caption for the image during training
        caption_idx = torch.randint(0, len(sample['captions']), (1,)).item()
        caption = sample['captions'][caption_idx]
        
        # Get image from MinIO
        try:
            img_obj = self.minio_client.get_object(
                self.bucket_name,
                f"images/{image_id}"
            )
            img_bytes = img_obj.read()
            image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            
            # Apply transformations
            if self.transform:
                image = self.transform(image)
            
            # Tokenize caption
            encoded_caption = self.tokenizer(
                caption,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            return {
                'image': image,
                'input_ids': encoded_caption['input_ids'].squeeze(),
                'attention_mask': encoded_caption['attention_mask'].squeeze(),
                'image_id': image_id,
                'caption': caption
            }
        
        except Exception as e:
            print(f"Error loading sample {image_id}: {str(e)}")
            # Return a dummy sample in case of error
            return self.__getitem__((idx + 1) % len(self))

def get_flickr8k_dataloaders(batch_size=32, num_workers=4):
    """
    Create dataloaders for Flickr8k dataset from MinIO
    
    Args:
        batch_size (int): Batch size for dataloaders
        num_workers (int): Number of workers for data loading
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = MinIOFlickr8kDataset(split='train')
    val_dataset = MinIOFlickr8kDataset(split='val')
    test_dataset = MinIOFlickr8kDataset(split='test')
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader

# Example usage
if __name__ == "__main__":
    # Set environment variables for MinIO connection
    os.environ['MINIO_ENDPOINT'] = 'localhost:9000'
    os.environ['MINIO_ACCESS_KEY'] = 'minioadmin'
    os.environ['MINIO_SECRET_KEY'] = 'minioadmin'
    os.environ['MINIO_BUCKET'] = 'flickr8k'
    
    # Create a small subset of the dataset for testing
    train_dataset = MinIOFlickr8kDataset(split='train')
    
    # Print some stats
    print(f"Dataset size: {len(train_dataset)}")
    
    # Test getting a sample
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        print(f"Sample image_id: {sample['image_id']}")
        print(f"Sample caption: {sample['caption']}")
        print(f"Image tensor shape: {sample['image'].shape}")
        print(f"Input IDs shape: {sample['input_ids'].shape}")
        print(f"Attention mask shape: {sample['attention_mask'].shape}") 