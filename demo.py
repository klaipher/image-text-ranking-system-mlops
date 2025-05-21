#!/usr/bin/env python3
import os
import requests
import json
import argparse
import base64
from PIL import Image
import io
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

API_URL = os.environ.get("API_URL", "http://localhost:8000")

def encode_image(image_path):
    """
    Encode an image using the API
    """
    # Open the file
    with open(image_path, 'rb') as f:
        files = {'image': (os.path.basename(image_path), f, 'image/jpeg')}
        response = requests.post(f"{API_URL}/encode_image", files=files)
    
    if response.status_code != 200:
        print(f"Error encoding image: {response.text}")
        return None
    
    return response.json()

def encode_text(text):
    """
    Encode a text query using the API
    """
    response = requests.post(
        f"{API_URL}/encode_text",
        data={'text': text}
    )
    
    if response.status_code != 200:
        print(f"Error encoding text: {response.text}")
        return None
    
    return response.json()

def search_images_by_text(text, image_data_list, top_k=5):
    """
    Search for images using a text query
    """
    response = requests.post(
        f"{API_URL}/search_images_by_text",
        data={
            'text': text,
            'image_embeddings': json.dumps(image_data_list),
            'top_k': top_k
        }
    )
    
    if response.status_code != 200:
        print(f"Error searching images: {response.text}")
        return None
    
    return response.json()

def main():
    parser = argparse.ArgumentParser(description='Demo for Image-Text Ranking API')
    parser.add_argument('--images_dir', type=str, default='data/images', 
                        help='Directory containing images to encode')
    parser.add_argument('--query', type=str, default='a dog running on the beach',
                        help='Text query for image search')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of top images to retrieve')
    args = parser.parse_args()
    
    # Check API health
    try:
        health_response = requests.get(f"{API_URL}/health")
        if health_response.status_code != 200:
            print(f"API is not healthy: {health_response.text}")
            return
        print("API is healthy and ready to use!")
    except Exception as e:
        print(f"Error connecting to API: {str(e)}")
        print(f"Make sure the API is running at {API_URL}")
        return
    
    # Get images from directory
    images_dir = Path(args.images_dir)
    if not images_dir.exists():
        print(f"Images directory {args.images_dir} does not exist")
        return
    
    # Encode images
    print(f"Encoding images from {args.images_dir}...")
    image_data_list = []
    for img_path in images_dir.glob('*.jpg'):
        print(f"Encoding {img_path.name}...")
        
        image_data = encode_image(str(img_path))
        if image_data:
            image_data['filename'] = img_path.name
            image_data['path'] = str(img_path)
            image_data_list.append(image_data)
    
    if not image_data_list:
        print("No images were successfully encoded")
        return
    
    print(f"Successfully encoded {len(image_data_list)} images")
    
    # Search images by text
    print(f"\nSearching for images matching: '{args.query}'")
    search_results = search_images_by_text(args.query, image_data_list, args.top_k)
    
    if not search_results:
        print("No search results returned")
        return
    
    # Display results
    plt.figure(figsize=(15, 8))
    plt.suptitle(f"Top {len(search_results['results'])} images for query: '{args.query}'", fontsize=16)
    
    for i, result in enumerate(search_results['results']):
        # Find the image path
        img_path = None
        for img_data in image_data_list:
            if img_data['filename'] == result['filename']:
                img_path = img_data['path']
                break
        
        if img_path:
            img = Image.open(img_path)
            plt.subplot(1, min(args.top_k, 5), i + 1)
            plt.imshow(np.array(img))
            plt.title(f"Similarity: {result['similarity']:.4f}")
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main() 