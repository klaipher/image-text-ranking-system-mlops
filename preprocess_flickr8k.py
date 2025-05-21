import os
import argparse
import pandas as pd
import shutil
import zipfile
import requests
from tqdm import tqdm
import glob
import re

def download_file(url, destination):
    """
    Download a file from a URL to a destination
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    
    with open(destination, 'wb') as file, tqdm(
            desc=destination,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
        for data in response.iter_content(block_size):
            size = file.write(data)
            bar.update(size)

def find_file(base_dir, pattern):
    """Find a file matching a pattern in any subdirectory"""
    matches = glob.glob(os.path.join(base_dir, "**", pattern), recursive=True)
    if matches:
        return matches[0]
    return None

def find_all_images(base_dir):
    """Find all image files recursively in a directory"""
    image_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, file))
    return image_files

def preprocess_flickr8k(args):
    """
    Preprocess the Flickr8K dataset
    """
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'Images'), exist_ok=True)
    
    # Download the dataset if needed
    if args.download:
        print("Downloading Flickr8K dataset...")
        
        # Download images
        images_zip = os.path.join(args.output_dir, 'Flickr8k_Dataset.zip')
        if not os.path.exists(images_zip):
            print("Downloading Flickr8k_Dataset.zip...")
            download_file(
                'https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip',
                images_zip
            )
        
        # Download text annotations
        text_zip = os.path.join(args.output_dir, 'Flickr8k_text.zip')
        if not os.path.exists(text_zip):
            print("Downloading Flickr8k_text.zip...")
            download_file(
                'https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip',
                text_zip
            )
        
        # Extract images
        if not os.path.exists(os.path.join(args.output_dir, 'Flickr8k_Dataset')):
            print("Extracting Flickr8k_Dataset.zip...")
            with zipfile.ZipFile(images_zip, 'r') as zip_ref:
                zip_ref.extractall(args.output_dir)
        
        # Extract text annotations
        if not os.path.exists(os.path.join(args.output_dir, 'Flickr8k_text')):
            print("Extracting Flickr8k_text.zip...")
            with zipfile.ZipFile(text_zip, 'r') as zip_ref:
                zip_ref.extractall(args.output_dir)
    
        # Find all image files recursively
        print("Searching for image files...")
        all_images = find_all_images(os.path.join(args.output_dir, 'Flickr8k_Dataset'))
        print(f"Found {len(all_images)} image files")
        
        if all_images:
            print("Copying images to Images directory...")
            for img_path in tqdm(all_images):
                img_file = os.path.basename(img_path)
                dst_path = os.path.join(args.output_dir, 'Images', img_file)
                if not os.path.exists(dst_path):
                    shutil.copy2(img_path, dst_path)
        else:
            print("No image files found. Manually searching for likely directories...")
            # List all directories just to check what was extracted
            for root, dirs, files in os.walk(args.output_dir):
                print(f"Directory: {root}")
                print(f"  Subdirectories: {dirs}")
                img_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if img_files:
                    print(f"  Contains {len(img_files)} image files")
                    # Copy these images
                    for img_file in tqdm(img_files):
                        src_path = os.path.join(root, img_file)
                        dst_path = os.path.join(args.output_dir, 'Images', img_file)
                        if not os.path.exists(dst_path):
                            shutil.copy2(src_path, dst_path)
    
    # List the images we found
    image_files = os.listdir(os.path.join(args.output_dir, 'Images'))
    image_files = [f for f in image_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Images in Images directory: {len(image_files)}")
    if image_files:
        print(f"Sample images: {image_files[:5]}")
    
    # Process the caption data
    print("Processing caption data...")
    
    # Find all text files to help diagnose the issue
    all_text_files = glob.glob(os.path.join(args.output_dir, "**", "*.txt"), recursive=True)
    print(f"Found {len(all_text_files)} text files:")
    for txt_file in all_text_files:
        print(f"  - {txt_file}")
    
    # Find the captions file
    captions_file = find_file(args.output_dir, "*token*.txt")
    if not captions_file:
        captions_file = os.path.join(args.output_dir, 'Flickr8k_text', 'Flickr8k.token.txt')
    
    if not os.path.exists(captions_file):
        print(f"Captions file not found at {captions_file}")
        
        # Try to determine the actual captions file by reading each text file
        found_caption_file = None
        for txt_file in all_text_files:
            if os.path.exists(txt_file):
                try:
                    with open(txt_file, 'r') as f:
                        first_lines = [f.readline().strip() for _ in range(5)]
                        # Look for patterns indicating captions: image filename patterns or #digit patterns
                        has_image_patterns = any(re.search(r'\d+\.jpg', line) for line in first_lines)
                        has_caption_markers = any('#' in line for line in first_lines)
                        if has_image_patterns or has_caption_markers:
                            print(f"Found likely captions file: {txt_file}")
                            print(f"First few lines: {first_lines}")
                            found_caption_file = txt_file
                            break
                except:
                    pass
        
        if found_caption_file:
            captions_file = found_caption_file
        else:
            print("\nCreating a dummy captions.csv using available images")
            # Create a minimal captions.csv from available images
            df = pd.DataFrame({
                'image': image_files,
                'caption': [f'Caption for {img}' for img in image_files],
                'split': ['train'] * len(image_files)  # All train for simplicity
            })
            output_file = os.path.join(args.output_dir, 'captions.csv')
            df.to_csv(output_file, sep='\t', index=False)
            print(f"Created {output_file} with {len(df)} entries")
            return
    
    print(f"Using captions file: {captions_file}")
    
    # Look for train/val/test splits
    train_file = find_file(args.output_dir, "*trainImages*.txt")
    val_file = find_file(args.output_dir, "*devImages*.txt") or find_file(args.output_dir, "*valImages*.txt")
    test_file = find_file(args.output_dir, "*testImages*.txt")
    
    if not train_file:
        train_file = os.path.join(args.output_dir, 'Flickr8k_text', 'Flickr_8k.trainImages.txt')
    if not val_file:
        val_file = os.path.join(args.output_dir, 'Flickr8k_text', 'Flickr_8k.devImages.txt')
    if not test_file:
        test_file = os.path.join(args.output_dir, 'Flickr8k_text', 'Flickr_8k.testImages.txt')
    
    # Read the image lists for each split
    train_images = set()
    val_images = set()
    test_images = set()
    
    # Try to read split files if they exist
    try:
        if os.path.exists(train_file):
            with open(train_file, 'r') as f:
                for line in f:
                    train_images.add(line.strip())
        
        if os.path.exists(val_file):
            with open(val_file, 'r') as f:
                for line in f:
                    val_images.add(line.strip())
        
        if os.path.exists(test_file):
            with open(test_file, 'r') as f:
                for line in f:
                    test_images.add(line.strip())
    except Exception as e:
        print(f"Error reading split files: {e}")
    
    # If we don't have enough split information, create random splits from available images
    if len(train_images) < 10 or len(val_images) < 10 or len(test_images) < 10:
        print("Creating random splits from available images...")
        import random
        random.shuffle(image_files)
        total = len(image_files)
        train_size = int(total * 0.7)
        val_size = int(total * 0.15)
        
        train_images = set(image_files[:train_size])
        val_images = set(image_files[train_size:train_size+val_size])
        test_images = set(image_files[train_size+val_size:])
        
        print(f"Split sizes - Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")
    
    # Parse captions
    image_captions = []
    
    # Try to determine the delimiter and format
    with open(captions_file, 'r') as f:
        first_line = f.readline().strip()
    
    # Determine the delimiter
    if '\t' in first_line:
        delimiter = '\t'
    elif '#' in first_line:
        # Format like: 123456.jpg#1  A caption text
        delimiter = None  # Special handling
    else:
        delimiter = ' '
    
    print(f"Using delimiter type: '{delimiter}' for captions file")
    
    with open(captions_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
                
            # Process based on determined format
            if delimiter == '\t':
                parts = line.split('\t')
                if len(parts) >= 2:
                    image_id, caption = parts[0], parts[1]
            elif delimiter is None:  # Special handling for formats like: 123456.jpg#1  A caption text
                match = re.match(r'([^\s]+)[\s\t]+(.+)', line)
                if match:
                    image_id, caption = match.groups()
                else:
                    continue
            else:
                # Try to split on first space
                parts = line.split(' ', 1)
                if len(parts) == 2:
                    image_id, caption = parts[0], parts[1]
                else:
                    continue
            
            # Extract image ID from formats like 123456.jpg#1
            if '#' in image_id:
                image_id = image_id.split('#')[0]  # Remove the #number suffix
            
            # Make sure it has a filename extension if missing
            if not any(image_id.lower().endswith(ext) for ext in ('.jpg', '.jpeg', '.png')):
                # Try to find a matching file in our image directory
                for img in image_files:
                    if img.startswith(image_id):
                        image_id = img
                        break
                else:
                    # If no match found, assume jpg
                    if not image_id.lower().endswith('.jpg'):
                        image_id = image_id + '.jpg'
            
            # Determine split - first check explicit splits
            if image_id in train_images:
                split = 'train'
            elif image_id in val_images:
                split = 'val'
            elif image_id in test_images:
                split = 'test'
            else:
                # If not in explicit splits but the file exists in our Images directory,
                # assign to train by default
                if image_id in image_files:
                    split = 'train'
                else:
                    continue  # Skip entries for missing images
            
            image_captions.append({
                'image': image_id,
                'caption': caption,
                'split': split
            })
    
    # Create a DataFrame
    df = pd.DataFrame(image_captions)
    
    if df.empty:
        print("Warning: No captions were parsed. Creating a dataset using available images.")
        # Create a dataset using available images
        df = pd.DataFrame({
            'image': image_files,
            'caption': [f'Caption for {img}' for img in image_files],
            'split': ['train'] * len(image_files)  # All train for simplicity
        })
    
    # Filter to only include images we actually have
    df = df[df['image'].isin(image_files)]
    
    if len(df) == 0:
        print("No matching images found between captions and image directory. Creating dummy data.")
        df = pd.DataFrame({
            'image': image_files,
            'caption': [f'Caption for {img}' for img in image_files],
            'split': ['train'] * len(image_files)  # All train for simplicity
        })
    
    # Save to TSV
    output_file = os.path.join(args.output_dir, 'captions.csv')
    df.to_csv(output_file, sep='\t', index=False)
    
    print(f"Dataset preprocessed and saved to {output_file}")
    print(f"Total images: {len(df['image'].unique())}")
    print(f"Total captions: {len(df)}")
    print(f"Train images: {len(df[df['split'] == 'train']['image'].unique())}")
    print(f"Val images: {len(df[df['split'] == 'val']['image'].unique())}")
    print(f"Test images: {len(df[df['split'] == 'test']['image'].unique())}")
    
    # Verify images are in the right place
    missing_images = []
    for img_id in df['image'].unique():
        img_path = os.path.join(args.output_dir, 'Images', img_id)
        if not os.path.exists(img_path):
            missing_images.append(img_id)
    
    if missing_images:
        print(f"Warning: {len(missing_images)} images referenced in captions are missing from the Images directory.")
        if args.verbose and missing_images:
            print("Missing images:", missing_images[:10], "..." if len(missing_images) > 10 else "")
            
            # For missing images, check if we can find them elsewhere
            found_elsewhere = 0
            for missing_img in missing_images[:10]:  # Just check the first few
                possible_matches = find_all_images(args.output_dir)
                matches = [p for p in possible_matches if os.path.basename(p) == missing_img]
                if matches:
                    print(f"Found {missing_img} at: {matches[0]}")
                    # Copy it to the Images directory
                    shutil.copy2(matches[0], os.path.join(args.output_dir, 'Images', missing_img))
                    found_elsewhere += 1
            
            if found_elsewhere > 0:
                print(f"Copied {found_elsewhere} missing images to the Images directory")
    else:
        print("All images referenced in captions were found in the Images directory.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess Flickr8K dataset')
    parser.add_argument('--output_dir', type=str, default='data/flickr8k', help='Output directory')
    parser.add_argument('--download', action='store_true', help='Download the dataset')
    parser.add_argument('--verbose', action='store_true', help='Print verbose information')
    
    args = parser.parse_args()
    preprocess_flickr8k(args) 