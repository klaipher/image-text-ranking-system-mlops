import os
import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

from model import ImageTextRankingModel
from data_flows.data_loader import Flickr8KDataset

def search_images(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    model = ImageTextRankingModel(embedding_dim=args.embedding_dim, temperature=args.temperature)
    
    # Check if model exists
    if os.path.exists(os.path.join(args.model_dir, 'best_model.pth')):
        checkpoint = torch.load(os.path.join(args.model_dir, 'best_model.pth'), map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint['epoch']} with validation loss {checkpoint['loss']:.4f}")
    else:
        print("No checkpoint found. Using untrained model for demonstration purposes.")
    
    model.to(device)
    model.eval()
    
    # Check if Images directory exists and has images
    images_dir = os.path.join(args.data_dir, 'Images')
    if not os.path.exists(images_dir) or not os.listdir(images_dir):
        print(f"Warning: No images found in {images_dir}")
        print("Creating a dummy image for demonstration")
        os.makedirs(images_dir, exist_ok=True)
        dummy_img_path = os.path.join(images_dir, 'dummy.jpg')
        img = Image.new('RGB', (224, 224), color='gray')
        img.save(dummy_img_path)
    
    # Check if captions file exists
    captions_file = os.path.join(args.data_dir, 'captions.csv')
    if not os.path.exists(captions_file):
        print(f"Warning: Captions file not found at {captions_file}")
        print("Creating a minimal captions file for demonstration")
        import pandas as pd
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        df = pd.DataFrame({
            'image': image_files,
            'caption': [f'Caption for {img}' for img in image_files],
            'split': ['test'] * len(image_files)
        })
        df.to_csv(captions_file, sep='\t', index=False)
    
    # Load image database for search
    print("Loading image database...")
    
    try:
        # Create a dataset for loading images
        split = 'test' if args.use_test_set else 'all'
        dataset = Flickr8KDataset(
            os.path.join(args.data_dir, 'Images'),
            os.path.join(args.data_dir, 'captions.csv'),
            split=split
        )
        
        # Make sure we have at least some images
        if len(dataset) == 0:
            print("No images found in dataset. Using all available images.")
            dataset = Flickr8KDataset(
                os.path.join(args.data_dir, 'Images'),
                os.path.join(args.data_dir, 'captions.csv'),
                split='all'  # Try to use all images regardless of split
            )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Creating an emergency dataset for demonstration")
        # Create a minimal dataset for demonstration
        dataset = Flickr8KDataset(
            os.path.join(args.data_dir, 'Images'),
            os.path.join(args.data_dir, 'captions.csv'),
            split='all'
        )
    
    print(f"Loaded {len(dataset)} images for search")
    
    # Prepare tokenizer
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Extract image embeddings for all images
    image_embeddings = []
    image_paths = []
    
    print("Computing image embeddings...")
    with torch.no_grad():
        for idx in range(len(dataset)):
            try:
                sample = dataset[idx]
                img = sample['image'].unsqueeze(0).to(device)
                img_path = sample['image_id']
                
                # Get image embedding
                img_embedding = model.get_image_embeddings(img)
                
                # Store embedding and path
                image_embeddings.append(img_embedding.cpu().numpy())
                image_paths.append(img_path)
            except Exception as e:
                print(f"Error processing image {idx}: {e}")
    
    if not image_embeddings:
        print("No valid images found. Cannot perform search.")
        return
    
    # Convert list of embeddings to a single array
    image_embeddings = np.vstack(image_embeddings)
    
    # Process the query
    print(f"Processing query: '{args.query}'")
    
    # Tokenize the query
    encoded_query = tokenizer(
        args.query,
        padding='max_length',
        truncation=True,
        max_length=64,
        return_tensors='pt'
    )
    
    input_ids = encoded_query['input_ids'].to(device)
    attention_mask = encoded_query['attention_mask'].to(device)
    
    # Get the text embedding
    with torch.no_grad():
        text_embedding = model.get_text_embeddings(input_ids, attention_mask)
        text_embedding = text_embedding.cpu().numpy()
    
    # Calculate similarity between the query and all images
    similarities = np.dot(text_embedding, image_embeddings.T)[0]
    
    # Sort by similarity
    top_k = min(args.top_k, len(image_paths))
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # Display results
    num_cols = min(3, top_k)
    num_rows = (top_k + num_cols - 1) // num_cols
    plt.figure(figsize=(15, 5 * num_rows))
    
    for i, idx in enumerate(top_indices):
        img_path = os.path.join(args.data_dir, 'Images', image_paths[idx])
        try:
            img = Image.open(img_path).convert('RGB')
            
            plt.subplot(num_rows, num_cols, i + 1)
            plt.imshow(img)
            plt.title(f"Rank {i+1}: {similarities[idx]:.4f}")
            plt.axis('off')
        except Exception as e:
            print(f"Error displaying image {img_path}: {e}")
    
    plt.suptitle(f"Top {top_k} images for query: '{args.query}'", fontsize=16)
    plt.tight_layout()
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, 'search_results.jpg')
    plt.savefig(results_path)
    print(f"Results saved to {results_path}")
    
    # Display if requested
    if args.show:
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Search images based on text query')
    parser.add_argument('--query', type=str, default="a beautiful landscape", help='Text query for image search')
    parser.add_argument('--data_dir', type=str, default='data/flickr8k', help='Path to data directory')
    parser.add_argument('--model_dir', type=str, default='models', help='Path to saved models')
    parser.add_argument('--embedding_dim', type=int, default=512, help='Dimension of the embedding space')
    parser.add_argument('--temperature', type=float, default=0.07, help='Temperature parameter for softmax')
    parser.add_argument('--top_k', type=int, default=9, help='Number of top images to retrieve')
    parser.add_argument('--use_test_set', action='store_true', help='Use test set for search')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--show', action='store_true', help='Show the results')
    
    args = parser.parse_args()
    search_images(args) 