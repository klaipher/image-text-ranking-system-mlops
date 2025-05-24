import os
import argparse
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import ImageTextRankingModel
from data_flows.data_loader import get_dataloader, prepare_flickr8k

def contrastive_loss(similarity, labels=None):
    """
    Compute the InfoNCE/NT-Xent loss.
    Args:
        similarity: cosine similarity matrix of shape (batch_size, batch_size)
        labels: optional labels for supervised contrastive learning
    """
    batch_size = similarity.shape[0]
    
    # If no labels provided, use the diagonal as positive pairs
    if labels is None:
        labels = torch.arange(batch_size, device=similarity.device)
        
    # Positive logits are on the diagonal
    logits_max, _ = torch.max(similarity, dim=1, keepdim=True)
    similarity = similarity - logits_max.detach()  # For numerical stability
    
    # Create a mask for positive pairs
    mask = torch.zeros_like(similarity)
    mask.scatter_(1, labels.view(-1, 1), 1)
    
    # Compute log_prob
    exp_logits = torch.exp(similarity)
    log_prob = similarity - torch.log(exp_logits.sum(dim=1, keepdim=True))
    
    # Compute mean of log-likelihood over positive pairs
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
    
    # Loss
    loss = -mean_log_prob_pos.mean()
    
    return loss

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model directory if it doesn't exist
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Initialize tensorboard
    writer = SummaryWriter(log_dir=os.path.join(args.model_dir, 'logs'))
    
    # Prepare dataset
    prepare_flickr8k(args.data_dir)
    
    # Get data loaders
    train_loader = get_dataloader(
        os.path.join(args.data_dir, 'Images'),
        os.path.join(args.data_dir, 'captions.csv'),
        split='train',
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    val_loader = get_dataloader(
        os.path.join(args.data_dir, 'Images'),
        os.path.join(args.data_dir, 'captions.csv'),
        split='val',
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Initialize model
    model = ImageTextRankingModel(embedding_dim=args.embedding_dim, temperature=args.temperature)
    model.to(device)
    
    # Initialize optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        with tqdm(train_loader, unit="batch") as t:
            t.set_description(f"Epoch {epoch+1}/{args.epochs}")
            
            for batch in t:
                images = batch['image'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                # Forward pass
                similarity = model(images, input_ids, attention_mask)
                
                # Compute loss
                loss = contrastive_loss(similarity)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update metrics
                train_loss += loss.item()
                t.set_postfix(loss=loss.item())
        
        # Calculate average training loss
        avg_train_loss = train_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                # Forward pass
                similarity = model(images, input_ids, attention_mask)
                
                # Compute loss
                loss = contrastive_loss(similarity)
                
                # Update metrics
                val_loss += loss.item()
        
        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_loader)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Update learning rate
        scheduler.step()
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, os.path.join(args.model_dir, 'best_model.pth'))
            print(f"Saved best model with val loss: {best_val_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
            }, os.path.join(args.model_dir, f'checkpoint_epoch_{epoch+1}.pth'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Image-Text Ranking Model')
    parser.add_argument('--data_dir', type=str, default='data/flickr8k', help='Path to data directory')
    parser.add_argument('--model_dir', type=str, default='models', help='Path to save models')
    parser.add_argument('--embedding_dim', type=int, default=512, help='Dimension of the embedding space')
    parser.add_argument('--temperature', type=float, default=0.07, help='Temperature parameter for softmax')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--save_every', type=int, default=5, help='Save model every n epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    
    args = parser.parse_args()
    train(args) 