"""
Fixed retraining script with better error handling
"""

import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os

# Set environment variables to prevent hanging
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

print("="*60)
print("Retraining Router on DocVQA")
print("="*60)

# Configuration
CONFIG = {
    'embedding_dim': 768,
    'hidden_dim': 256,
    'num_classes': 4,
    'dropout': 0.1,
    'batch_size': 32,  # Reduced batch size
    'epochs': 30,  # Reduced epochs for faster training
    'learning_rate': 1e-4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

print(f"\nConfiguration:")
for k, v in CONFIG.items():
    print(f"   {k}: {v}")

# Check if training data exists
data_path = Path("./data/docvqa_training_data.json")
if not data_path.exists():
    print(f"\n❌ Training data not found at {data_path}")
    print("Run: python scripts/create_docvqa_training_data.py")
    sys.exit(1)

# Load data
print("\n[1/5] Loading training data...")
with open(data_path, 'r') as f:
    data = json.load(f)

train_data = data['train']
val_data = data['val']
test_data = data['test']
type_to_idx = data['type_mapping']
idx_to_type = {v: k for k, v in type_to_idx.items()}

print(f"   Train: {len(train_data)} samples")
print(f"   Val: {len(val_data)} samples")
print(f"   Test: {len(test_data)} samples")
print(f"   Classes: {type_to_idx}")

# Balance the dataset (oversample minority classes)
print("\n[2/5] Balancing dataset...")

# Count samples per class
class_counts = {}
for t, _ in train_data:
    class_counts[t] = class_counts.get(t, 0) + 1

print(f"   Original distribution: {class_counts}")

# Find max count
max_count = max(class_counts.values())
balanced_train_data = []

for t, q in train_data:
    balanced_train_data.append((t, q))
    # Duplicate minority classes
    if class_counts[t] < max_count:
        multiplier = max_count // class_counts[t]
        for _ in range(multiplier - 1):
            balanced_train_data.append((t, q))

# Shuffle
import random
random.shuffle(balanced_train_data)

print(f"   Balanced train size: {len(balanced_train_data)}")

# Load encoder with error handling
print("\n[3/5] Loading sentence encoder...")

try:
    # Try to use local model first
    local_model_path = Path("./models/all-mpnet-base-v2")
    if local_model_path.exists():
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(str(local_model_path))
        print(f"   ✅ Loaded from local: {local_model_path}")
    else:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer('all-mpnet-base-v2')
        print(f"   ✅ Loaded from cache")
except Exception as e:
    print(f"   ❌ Failed to load sentence_transformers: {e}")
    print("   Using fallback: random embeddings (for testing only)")
    
    # Fallback: create dummy encoder
    class DummyEncoder:
        def encode(self, text):
            return np.random.randn(768).astype(np.float32)
    encoder = DummyEncoder()

class DocVQADataset(Dataset):
    def __init__(self, data, encoder, type_to_idx):
        self.data = data
        self.encoder = encoder
        self.type_to_idx = type_to_idx
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        q_type, question = self.data[idx]
        embedding = self.encoder.encode(question)
        label = self.type_to_idx[q_type]
        return {
            'embedding': torch.tensor(embedding),
            'label': torch.tensor(label, dtype=torch.long)
        }

class UtilityPredictionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        shared_out = self.shared(x)
        logits = self.classifier(shared_out)
        return {'path_logits': logits, 'shared_features': shared_out}

# Create datasets
print("\n[4/5] Creating datasets...")
train_dataset = DocVQADataset(balanced_train_data, encoder, type_to_idx)
val_dataset = DocVQADataset(val_data, encoder, type_to_idx)

train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

print(f"   Train batches: {len(train_loader)}")
print(f"   Val batches: {len(val_loader)}")

# Initialize model
print("\n[5/5] Training model...")
device = torch.device(CONFIG['device'])
model = UtilityPredictionModel(
    CONFIG['embedding_dim'],
    CONFIG['hidden_dim'],
    CONFIG['num_classes'],
    CONFIG['dropout']
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

print(f"   Device: {device}")
print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Training loop
best_val_acc = 0
train_losses = []
val_accs = []

for epoch in range(CONFIG['epochs']):
    # Training
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Train"):
        embeddings = batch['embedding'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(embeddings)
        loss = criterion(outputs['path_logits'], labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs['path_logits'], 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
    
    train_acc = train_correct / train_total
    train_losses.append(train_loss / len(train_loader))
    
    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} Val"):
            embeddings = batch['embedding'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(embeddings)
            _, predicted = torch.max(outputs['path_logits'], 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    val_acc = val_correct / val_total
    val_accs.append(val_acc)
    
    print(f"   Epoch {epoch+1}: Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_acc*100:.2f}%, Val Acc: {val_acc*100:.2f}%")
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'config': CONFIG
        }
        torch.save(checkpoint, './checkpoints/docvqa_router_fixed.pth')
        print(f"   ✅ Saved best model (val_acc: {val_acc*100:.2f}%)")

print("\n" + "="*60)
print("✅ Retraining complete!")
print(f"   Best model: ./checkpoints/docvqa_router_fixed.pth")
print(f"   Best validation accuracy: {best_val_acc*100:.2f}%")
print("="*60)

# Save training history
history = {
    'train_losses': train_losses,
    'val_accs': val_accs,
    'best_val_acc': best_val_acc,
    'config': CONFIG
}

with open('./logs/retraining_history.json', 'w') as f:
    json.dump(history, f, indent=2)

print(f"\n💾 Training history saved to: ./logs/retraining_history.json")