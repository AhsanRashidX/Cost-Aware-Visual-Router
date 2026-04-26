# train_real_router.py
"""
Train your router on the real multimodal training data
"""

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torch
from torch.utils.data import DataLoader, random_split
from train_router_working import RouterDataset, train_router
import json
from pathlib import Path

def train_on_real_data():
    # Load the data you just generated
    dataset = RouterDataset('./strong_training_data')
    print(f"Loaded {len(dataset)} training samples")
    
    # Split train/val
    val_size = int(len(dataset) * 0.1)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    config = {
        'embedding_dim': 768,
        'hidden_dim': 256,
        'num_paths': 4,
        'dropout': 0.1,
        'num_epochs': 100,
        'lr': 1e-4,
        'weight_decay': 1e-5,
    }
    # Add this after config definition, before train_router() call
    # Create TensorBoard writer
    log_dir = f"./runs/router_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir)
    print(f"📊 TensorBoard logs will be saved to: {log_dir}")
    # Train the model
    model = train_router(config, train_loader, val_loader)
    
    # Save the trained model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config
    }, './checkpoints/real_router_final.pth')
    
    print("✅ Model trained and saved!")

if __name__ == '__main__':
    train_on_real_data()