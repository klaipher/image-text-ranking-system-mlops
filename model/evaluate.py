import os
import argparse
import torch
import numpy as np
from tqdm import tqdm

from model import ImageTextRankingModel
from data_flows.data_loader import get_dataloader

def compute_metrics(similarities, labels=None):
    """
    Compute ranking metrics
    """
    if labels is None:
        # Use diagonal as ground truth
        labels = torch.arange(similarities.size(0), device=similarities.device)
    
    # For each text query, compute recall@k for retrieving images
    recalls = {}
    for k in [1, 5, 10]:
        # Get top k predictions for each query
        _, indices = similarities.topk(k, dim=1)
        # Check if labels are in top k predictions
        correct = torch.zeros(similarities.size(0), dtype=torch.bool, device=similarities.device)
        for i in range(similarities.size(0)):
            correct[i] = labels[i] in indices[i]
        # Compute recall@k
        recall = correct.float().mean().item()
        recalls[f'R@{k}'] = recall * 100
    
    # Mean rank of the ground truth
    ranks = torch.zeros(similarities.size(0), dtype=torch.long, device=similarities.device)
    for i in range(similarities.size(0)):
        # Get the rank of the ground truth
        rank = torch.where(similarities[i].argsort(descending=True) == labels[i])[0].item() + 1
        ranks[i] = rank
    # Compute mean rank
    mean_rank = ranks.float().mean().item()
    # Compute median rank
    median_rank = torch.median(ranks.float()).item()
    
    return {
        **recalls,
        'Mean Rank': mean_rank,
        'Median Rank': median_rank
    }

def evaluate(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    model = ImageTextRankingModel(embedding_dim=args.embedding_dim, temperature=args.temperature)
    
    # Load checkpoint
    if os.path.exists(os.path.join(args.model_dir, 'best_model.pth')):
        checkpoint = torch.load(os.path.join(args.model_dir, 'best_model.pth'), map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint['epoch']} with validation loss {checkpoint['loss']:.4f}")
    else:
        print("No checkpoint found. Starting from scratch.")
    
    model.to(device)
    model.eval()
    
    # Get test dataloader
    test_loader = get_dataloader(
        os.path.join(args.data_dir, 'Images'),
        os.path.join(args.data_dir, 'captions.csv'),
        split='test',
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    all_metrics = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass
            similarity = model(images, input_ids, attention_mask)
            
            # Compute metrics
            metrics = compute_metrics(similarity)
            all_metrics.append(metrics)
    
    # Average metrics across all batches
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    
    # Print results
    print("Evaluation Results:")
    for k, v in avg_metrics.items():
        print(f"{k}: {v:.2f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Image-Text Ranking Model')
    parser.add_argument('--data_dir', type=str, default='data/flickr8k', help='Path to data directory')
    parser.add_argument('--model_dir', type=str, default='models', help='Path to saved models')
    parser.add_argument('--embedding_dim', type=int, default=512, help='Dimension of the embedding space')
    parser.add_argument('--temperature', type=float, default=0.07, help='Temperature parameter for softmax')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    
    args = parser.parse_args()
    evaluate(args) 