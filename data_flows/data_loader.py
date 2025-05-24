import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoTokenizer
from torchvision import transforms
import logging

class Flickr8KDataset(Dataset):
    def __init__(self, data_path, captions_file, split='train', transform=None, max_length=64):
        """
        Args:
            data_path (str): Path to the Flickr8K images
            captions_file (str): Path to the captions file
            split (str): 'train', 'val', or 'test'
            transform: Image transforms
            max_length (int): Maximum length of captions
        """
        self.data_path = data_path
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load captions data
        try:
            self.df = pd.read_csv(captions_file, delimiter='\t')
            
            # Filter by split
            if split == 'train':
                self.df = self.df[self.df['split'] == 'train']
            elif split == 'val':
                self.df = self.df[self.df['split'] == 'val']
            elif split == 'test':
                self.df = self.df[self.df['split'] == 'test']
            
            # Verify images exist
            valid_images = []
            for idx, row in self.df.iterrows():
                img_path = os.path.join(self.data_path, row['image'])
                if os.path.exists(img_path):
                    valid_images.append(idx)
                
            if len(valid_images) < len(self.df):
                print(f"Warning: {len(self.df) - len(valid_images)} out of {len(self.df)} image entries don't exist in {self.data_path}")
                self.df = self.df.iloc[valid_images].reset_index(drop=True)
                
            if len(self.df) == 0:
                print(f"Warning: No valid images found for split '{split}'. Creating dummy data.")
                # Create dummy data if no valid images found
                self._create_dummy_data()
                
        except Exception as e:
            print(f"Error loading captions file: {e}")
            self._create_dummy_data()
            
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        self.max_length = max_length
    
    def _create_dummy_data(self):
        """Create dummy data for testing"""
        print("Creating dummy dataset for testing purposes")
        # Find available images
        if os.path.exists(self.data_path) and os.listdir(self.data_path):
            available_images = [f for f in os.listdir(self.data_path) 
                               if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:10]
            if not available_images:
                # No images found, create placeholder
                self.df = pd.DataFrame({
                    'image': ['dummy.jpg'],
                    'caption': ['This is a dummy caption.'],
                    'split': ['train']
                })
                # Create a dummy image
                dummy_img_path = os.path.join(self.data_path, 'dummy.jpg')
                if not os.path.exists(dummy_img_path):
                    try:
                        img = Image.new('RGB', (224, 224), color='gray')
                        img.save(dummy_img_path)
                    except:
                        pass
            else:
                # Use available images
                self.df = pd.DataFrame({
                    'image': available_images,
                    'caption': [f'Caption for {img}' for img in available_images],
                    'split': ['train'] * len(available_images)
                })
        else:
            # No images directory
            os.makedirs(self.data_path, exist_ok=True)
            self.df = pd.DataFrame({
                'image': ['dummy.jpg'],
                'caption': ['This is a dummy caption.'],
                'split': ['train']
            })
            # Create a dummy image
            dummy_img_path = os.path.join(self.data_path, 'dummy.jpg')
            if not os.path.exists(dummy_img_path):
                try:
                    img = Image.new('RGB', (224, 224), color='gray')
                    img.save(dummy_img_path)
                except:
                    pass
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Get image and caption
        img_name = self.df.iloc[idx]['image']
        caption = self.df.iloc[idx]['caption']
        
        # Load and transform image
        img_path = os.path.join(self.data_path, img_name)
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy image
            image = torch.zeros((3, 224, 224))
            
        # Tokenize caption
        try:
            encoded_caption = self.tokenizer(
                caption,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
        except Exception as e:
            print(f"Error tokenizing caption: {e}")
            # Return dummy tokenized output
            encoded_caption = {
                'input_ids': torch.zeros((1, self.max_length), dtype=torch.long),
                'attention_mask': torch.zeros((1, self.max_length), dtype=torch.long)
            }
        
        return {
            'image': image,
            'input_ids': encoded_caption['input_ids'].squeeze(),
            'attention_mask': encoded_caption['attention_mask'].squeeze(),
            'caption': caption,
            'image_id': img_name
        }

def get_dataloader(data_path, captions_file, split='train', batch_size=32, shuffle=True, num_workers=4):
    """
    Create a DataLoader for the Flickr8K dataset
    """
    try:
        dataset = Flickr8KDataset(data_path, captions_file, split)
        
        # Adjust num_workers based on system
        if os.name == 'posix' and os.uname().sysname == 'Darwin':  # macOS
            # Apple Silicon often has issues with multiprocessing
            if hasattr(os, 'uname') and 'arm64' in os.uname().machine:
                num_workers = min(2, num_workers)  # Limit workers on Apple Silicon
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=False
        )
    except Exception as e:
        print(f"Error creating dataloader: {e}")
        # Return a minimal dataloader for testing
        dummy_dataset = Flickr8KDataset(data_path, captions_file, split)
        return DataLoader(
            dummy_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0  # No multiprocessing for safety
        )

def prepare_flickr8k(data_dir):
    """Helper function to prepare Flickr8K dataset structure if needed"""
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'Images'), exist_ok=True)
    
    # Example format for captions file (create manually or download)
    if not os.path.exists(os.path.join(data_dir, 'captions.csv')):
        print(f"Please download Flickr8K dataset and place images in {data_dir}/Images/")
        print(f"Create a captions.csv file with format: image, caption, split")
        # This would be handled manually by downloading the dataset 