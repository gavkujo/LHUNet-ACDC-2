import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ACDCDataset  # Custom dataset class
from model import LHUNet  # LHU-Net model implementation
from tqdm import tqdm
import argparse

# Argument parser for hyperparameters
def get_args():
    parser = argparse.ArgumentParser(description='Train LHU-Net on ACDC dataset')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to train on')
    parser.add_argument('--data_path', type=str, default='./data/', help='Path to ACDC dataset')
    parser.add_argument('--save_path', type=str, default='./checkpoints/', help='Path to save model checkpoints')
    return parser.parse_args()

def train():
    args = get_args()
    
    # Load dataset
    train_dataset = ACDCDataset(args.data_path, train=True)
    val_dataset = ACDCDataset(args.data_path, train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Initialize model
    model = LHUNet(in_channels=1, out_channels=1).to(args.device)
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}', unit='batch') as pbar:
            for images, masks in pbar:
                images, masks = images.to(args.device), masks.to(args.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
        
        # Validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(args.device), masks.to(args.device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        
        print(f'Epoch {epoch+1}, Training Loss: {running_loss / len(train_loader)}, Validation Loss: {val_loss / len(val_loader)}')
        
        # Save model checkpoint
        torch.save(model.state_dict(), f'{args.save_path}/lhunet_epoch{epoch+1}.pth')

if __name__ == '__main__':
    train()
