#!/usr/bin/env python3
"""
Demo script for image-text ranking inference.
"""

import argparse
from pathlib import Path

from src.inference import create_predictor
from src.config import data_config


def demo_text_to_image_search(predictor, image_dir: Path, text_query: str, top_k: int = 5):
    """Demo text-to-image search."""
    print(f"\n🔍 Searching for: '{text_query}'")
    print("=" * 50)
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(list(image_dir.glob(f"*{ext}")))
        image_paths.extend(list(image_dir.glob(f"*{ext.upper()}")))
    
    if not image_paths:
        print("No images found in the directory!")
        return
    
    print(f"Found {len(image_paths)} images")
    
    # Convert to strings
    image_paths = [str(p) for p in image_paths]
    
    # Rank images
    results = predictor.rank_images(text_query, image_paths, top_k=top_k)
    
    print(f"\nTop {len(results)} results:")
    for result in results:
        print(f"  {result['rank']}. {Path(result['image_path']).name} (score: {result['score']:.3f})")


def demo_image_to_text_search(predictor, image_path: str, captions: list, top_k: int = 5):
    """Demo image-to-text search."""
    print(f"\n🖼️ Finding best captions for: {Path(image_path).name}")
    print("=" * 50)
    
    results = predictor.rank_texts(image_path, captions, top_k=top_k)
    
    print(f"\nTop {len(results)} captions:")
    for result in results:
        print(f"  {result['rank']}. \"{result['text']}\" (score: {result['score']:.3f})")


def demo_database_search(predictor, image_dir: Path, text_queries: list):
    """Demo database search functionality."""
    print(f"\n🗄️ Building image database from {image_dir}")
    print("=" * 50)
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(list(image_dir.glob(f"*{ext}")))
        image_paths.extend(list(image_dir.glob(f"*{ext.upper()}")))
    
    if not image_paths:
        print("No images found in the directory!")
        return
    
    # Convert to strings
    image_paths = [str(p) for p in image_paths]
    
    # Build database
    predictor.build_image_database(image_paths)
    
    # Search with multiple queries
    for query in text_queries:
        print(f"\n🔍 Database search for: '{query}'")
        results = predictor.search_database(query, top_k=3)
        
        for result in results:
            print(f"  {result['rank']}. {Path(result['image_path']).name} (score: {result['score']:.3f})")


def main():
    parser = argparse.ArgumentParser(description="Demo image-text ranking inference")
    parser.add_argument("--model-path", type=str, help="Path to trained model")
    parser.add_argument("--vocab-path", type=str, help="Path to vocabulary file")
    parser.add_argument("--image-dir", type=str, help="Directory containing images for search")
    parser.add_argument("--image-path", type=str, help="Single image path for image-to-text demo")
    parser.add_argument("--device", type=str, choices=['cpu', 'mps', 'cuda'], help="Device to use")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("IMAGE-TEXT RANKING INFERENCE DEMO")
    print("=" * 60)
    
    try:
        # Create predictor
        print("Loading model and vocabulary...")
        predictor = create_predictor(
            model_path=args.model_path,
            vocab_path=args.vocab_path,
            device=args.device
        )
        print("✅ Model loaded successfully!")
        
        # Demo 1: Text-to-image search
        if args.image_dir:
            image_dir = Path(args.image_dir)
            if image_dir.exists():
                demo_queries = [
                    "a dog playing in the park",
                    "people walking on the street",
                    "a beautiful sunset",
                    "children playing football",
                    "a cat sitting on a chair"
                ]
                
                for query in demo_queries[:2]:  # Show first 2 queries
                    demo_text_to_image_search(predictor, image_dir, query)
                
                # Demo database functionality
                demo_database_search(predictor, image_dir, demo_queries[2:4])
            else:
                print(f"Image directory {image_dir} does not exist!")
        
        # Demo 2: Image-to-text search  
        if args.image_path:
            image_path = Path(args.image_path)
            if image_path.exists():
                sample_captions = [
                    "a dog running in the grass",
                    "people walking down a street",
                    "a beautiful landscape with mountains",
                    "children playing in a playground",
                    "a cat lying on a sofa",
                    "cars parked on the road",
                    "a person riding a bicycle",
                    "birds flying in the sky",
                    "flowers blooming in a garden",
                    "a sunset over the ocean"
                ]
                
                demo_image_to_text_search(predictor, str(image_path), sample_captions)
            else:
                print(f"Image path {image_path} does not exist!")
        
        # Demo 3: Basic encoding functionality
        print(f"\n🧠 Testing basic encoding functionality")
        print("=" * 50)
        
        # Test text encoding
        sample_texts = ["a dog playing", "beautiful sunset", "people walking"]
        text_embeddings = predictor.encode_text(sample_texts)
        print(f"Text embeddings shape: {text_embeddings.shape}")
        
        if args.image_dir:
            image_dir = Path(args.image_dir)
            if image_dir.exists():
                # Test image encoding with first few images
                image_files = list(image_dir.glob("*.jpg"))[:3]
                if image_files:
                    image_embeddings = predictor.encode_image([str(f) for f in image_files])
                    print(f"Image embeddings shape: {image_embeddings.shape}")
                
                # Test similarity computation
                if len(image_files) > 0:
                    similarities = predictor.compute_similarity(text_embeddings, image_embeddings)
                    print(f"Similarity matrix shape: {similarities.shape}")
                    print(f"Sample similarities: {similarities[0]}")
        
        print(f"\n✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        raise


def quick_demo():
    """Quick demo with default paths."""
    print("Running quick demo with default paths...")
    
    # Check if data exists
    data_dir = data_config.processed_data_path
    models_dir = data_config.models_path
    
    if not (data_dir / "vocabulary.json").exists():
        print("❌ Vocabulary file not found. Please train the model first.")
        print("Run: python train.py")
        return
    
    if not (models_dir / "final_model.pt").exists():
        print("❌ Trained model not found. Please train the model first.")
        print("Run: python train.py")
        return
    
    # Create predictor with default paths
    predictor = create_predictor()
    
    # Test basic functionality
    print("Testing text encoding...")
    sample_texts = ["a dog playing in the park", "beautiful mountain landscape", "people walking"]
    text_embeddings = predictor.encode_text(sample_texts)
    print(f"✅ Text embeddings shape: {text_embeddings.shape}")
    
    # Check if sample images exist
    images_dir = data_dir / "images"
    if images_dir.exists():
        image_files = list(images_dir.glob("*.jpg"))[:3]
        if image_files:
            print("Testing image encoding...")
            image_embeddings = predictor.encode_image([str(f) for f in image_files])
            print(f"✅ Image embeddings shape: {image_embeddings.shape}")
            
            print("Testing similarity computation...")
            similarities = predictor.compute_similarity(text_embeddings, image_embeddings)
            print(f"✅ Similarity computation successful: {similarities.shape}")
    
    print("✅ Quick demo completed!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 1:
        # No arguments provided, run quick demo
        quick_demo()
    else:
        # Arguments provided, run full demo
        main() 