"""
Dataset module for Flickr8K image-text pairs.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import torchvision.transforms as transforms
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..config import data_config, model_config


class TextProcessor:
    """Simple text processor for captions."""
    
    def __init__(self, vocab_size: int = 10000, max_length: int = 64):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.word_to_idx = {'<pad>': 0, '<unk>': 1, '<start>': 2, '<end>': 3}
        self.idx_to_word = {0: '<pad>', 1: '<unk>', 2: '<start>', 3: '<end>'}
        self.vocab_built = False
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        text = text.lower().strip()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def build_vocabulary(self, texts: List[str]) -> None:
        """Build vocabulary from texts."""
        word_counts = {}
        
        for text in texts:
            words = self.clean_text(text).split()
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Sort by frequency and take top vocab_size - 4 (reserved tokens)
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        top_words = sorted_words[:self.vocab_size - 4]
        
        for i, (word, _) in enumerate(top_words):
            idx = i + 4  # Start after reserved tokens
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word
        
        self.vocab_built = True
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        if not self.vocab_built:
            raise ValueError("Vocabulary not built. Call build_vocabulary first.")
        
        words = self.clean_text(text).split()
        tokens = [self.word_to_idx['<start>']]
        
        for word in words[:self.max_length - 2]:  # Leave space for start/end tokens
            tokens.append(self.word_to_idx.get(word, self.word_to_idx['<unk>']))
        
        tokens.append(self.word_to_idx['<end>'])
        
        # Pad to max_length
        while len(tokens) < self.max_length:
            tokens.append(self.word_to_idx['<pad>'])
        
        return tokens[:self.max_length]
    
    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to text."""
        words = []
        for token in tokens:
            if token == self.word_to_idx['<pad>']:
                break
            if token in self.idx_to_word:
                word = self.idx_to_word[token]
                if word not in ['<start>', '<end>']:
                    words.append(word)
        return ' '.join(words)


class Flickr8kDataset(Dataset):
    """Dataset class for Flickr8K image-text pairs."""
    
    def __init__(
        self, 
        data_dir: Path,
        split: str = 'train',
        transform: Optional[transforms.Compose] = None,
        text_processor: Optional[TextProcessor] = None
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.text_processor = text_processor
        
        # Load data
        self.data = self._load_data()
        
    def _load_data(self) -> List[Dict]:
        """Load the dataset splits."""
        split_file = self.data_dir / f"{self.split}_data.json"
        
        if not split_file.exists():
            raise FileNotFoundError(f"Split file {split_file} not found. Run prepare_data first.")
        
        with open(split_file, 'r') as f:
            data = json.load(f)
        
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, str]:
        item = self.data[idx]
        
        # Load image
        image_path = self.data_dir / "images" / item['image_file']
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Process text
        caption = item['caption']
        if self.text_processor:
            text_tokens = torch.tensor(self.text_processor.encode(caption), dtype=torch.long)
        else:
            text_tokens = caption
        
        return image, text_tokens, item['image_file'], caption


class Flickr8kDataLoader:
    """Data loader factory for Flickr8K dataset."""
    
    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir) if data_dir else data_config.processed_data_path
        self.text_processor = TextProcessor(
            vocab_size=model_config.vocab_size,
            max_length=model_config.max_text_length
        )
        
        # Image transforms
        self.train_transform = transforms.Compose([
            transforms.Resize(model_config.image_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=model_config.image_mean, std=model_config.image_std)
        ])
        
        self.eval_transform = transforms.Compose([
            transforms.Resize(model_config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=model_config.image_mean, std=model_config.image_std)
        ])
        
    def prepare_data(self) -> None:
        """Download and prepare Flickr8K dataset."""
        print("Preparing Flickr8K dataset...")
        
        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        images_dir = self.data_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        # Download from Hugging Face if not exists
        if not (self.data_dir / "captions.txt").exists():
            self._download_flickr8k_hf()
        
        # Process captions and create splits
        self._process_captions()
        print("Dataset preparation complete!")
    
    def _download_flickr8k_hf(self) -> None:
        """Download Flickr8K dataset from Hugging Face."""
        print("Downloading Flickr8K dataset from Hugging Face...")
        
        try:
            # Load dataset from Hugging Face
            # Using jxie/flickr8k which is a well-maintained Flickr8K dataset
            dataset = load_dataset("jxie/flickr8k", split="train")
            
            print(f"Downloaded dataset with {len(dataset)} samples")
            
            # Create images directory
            images_dir = self.data_dir / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            
            # Prepare captions data
            captions_data = []
            
            print("Processing and saving images...")
            for i, sample in enumerate(tqdm(dataset)):
                # Save image
                image = sample['image']
                image_filename = f"image_{i:06d}.jpg"
                image_path = images_dir / image_filename
                
                # Convert to RGB if not already
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                image.save(image_path)
                
                # Collect all captions (caption_0 to caption_4)
                captions = []
                for j in range(5):
                    cap_key = f'caption_{j}'
                    if cap_key in sample and sample[cap_key] and sample[cap_key].strip():
                        captions.append(sample[cap_key].strip())
                
                # Add to captions data
                for caption in captions:
                    captions_data.append({
                        'image': image_filename,
                        'caption': caption
                    })
            
            # Save captions as CSV
            import pandas as pd
            df = pd.DataFrame(captions_data)
            captions_file = self.data_dir / "captions.txt"
            df.to_csv(captions_file, index=False)
            
            print(f"Dataset downloaded successfully!")
            print(f"Images saved to: {images_dir}")
            print(f"Captions saved to: {captions_file}")
            print(f"Total image-caption pairs: {len(captions_data)}")
            
        except Exception as e:
            print(f"Error downloading dataset from Hugging Face: {e}")
            print("Available alternatives:")
            print("1. Try a different Flickr8K dataset from Hugging Face")
            print("2. Download manually and place in data/processed/")
            print("   Required structure:")
            print("   data/processed/")
            print("   ├── images/")
            print("   │   ├── image_000001.jpg")
            print("   │   └── ...")
            print("   └── captions.txt")
            raise
    
    def _process_captions(self) -> None:
        """Process captions and create train/val/test splits."""
        captions_file = self.data_dir / "captions.txt"
        
        if not captions_file.exists():
            raise FileNotFoundError(
                f"Captions file not found at {captions_file}. "
                "Please run prepare_data() first to download the dataset."
            )
        
        # Read captions
        df = pd.read_csv(captions_file)
        
        # Group by image
        grouped = df.groupby('image')['caption'].apply(list).reset_index()
        
        # Create splits
        n_images = len(grouped)
        n_train = int(n_images * data_config.train_split)
        n_val = int(n_images * data_config.val_split)
        
        train_images = grouped[:n_train]
        val_images = grouped[n_train:n_train + n_val]
        test_images = grouped[n_train + n_val:]
        
        # Build vocabulary from training captions
        all_train_captions = []
        for _, row in train_images.iterrows():
            all_train_captions.extend(row['caption'])
        
        self.text_processor.build_vocabulary(all_train_captions)
        
        # Save vocabulary
        vocab_file = self.data_dir / "vocabulary.json"
        with open(vocab_file, 'w') as f:
            json.dump({
                'word_to_idx': self.text_processor.word_to_idx,
                'idx_to_word': self.text_processor.idx_to_word
            }, f)
        
        # Create split files
        for split_name, split_data in [('train', train_images), ('val', val_images), ('test', test_images)]:
            split_items = []
            for _, row in split_data.iterrows():
                for caption in row['caption']:
                    split_items.append({
                        'image_file': row['image'],
                        'caption': caption
                    })
            
            split_file = self.data_dir / f"{split_name}_data.json"
            with open(split_file, 'w') as f:
                json.dump(split_items, f)
        
        print(f"Created splits: train={len(train_images)}, val={len(val_images)}, test={len(test_images)} images")
    
    def load_text_processor(self) -> TextProcessor:
        """Load the text processor with built vocabulary."""
        vocab_file = self.data_dir / "vocabulary.json"
        
        if not vocab_file.exists():
            raise FileNotFoundError("Vocabulary file not found. Run prepare_data first.")
        
        with open(vocab_file, 'r') as f:
            vocab_data = json.load(f)
        
        self.text_processor.word_to_idx = {k: int(v) for k, v in vocab_data['word_to_idx'].items()}
        self.text_processor.idx_to_word = {int(k): v for k, v in vocab_data['idx_to_word'].items()}
        self.text_processor.vocab_built = True
        
        return self.text_processor
    
    def get_dataloader(self, split: str = 'train', batch_size: int = None, shuffle: bool = None) -> DataLoader:
        """Get data loader for specified split."""
        if batch_size is None:
            batch_size = model_config.batch_size
        
        if shuffle is None:
            shuffle = (split == 'train')
        
        # Load text processor
        text_processor = self.load_text_processor()
        
        # Select transform
        transform = self.train_transform if split == 'train' else self.eval_transform
        
        # Create dataset
        dataset = Flickr8kDataset(
            data_dir=self.data_dir,
            split=split,
            transform=transform,
            text_processor=text_processor
        )
        
        # Create dataloader
        import torch
        device = torch.device(model_config.device)
        pin_memory = False if device.type == "mps" else True

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=2,
            pin_memory=pin_memory
        )
        
        return dataloader


def get_dataloaders(data_dir: str = None) -> Dict[str, DataLoader]:
    """Get all data loaders."""
    loader_factory = Flickr8kDataLoader(data_dir)
    
    # Prepare data if needed
    if not (loader_factory.data_dir / "train_data.json").exists():
        loader_factory.prepare_data()
    
    return {
        'train': loader_factory.get_dataloader('train'),
        'val': loader_factory.get_dataloader('val'),
        'test': loader_factory.get_dataloader('test')
    } 