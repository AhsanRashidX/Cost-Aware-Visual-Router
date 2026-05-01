"""
Train router on truly balanced dataset
"""

import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

print("="*70)
print("TRAINING BALANCED ROUTER V2")
print("="*70)

# Configuration
CONFIG = {
    'embedding_dim': 768,
    'hidden_dim': 256,
    'num_classes': 3,
    'dropout': 0.1,
    'batch_size': 32,  # Smaller batch since dataset is smaller
    'epochs': 50,
    'learning_rate': 1e-4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

class BalancedRouterModel(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_classes=3, dropout=0.1):
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
        return {'path_logits': logits}

class BalancedDataset(Dataset):
    def __init__(self, data, tokenizer, embedding_model, device):
        self.data = data
        self.tokenizer = tokenizer
        self.embedding_model = embedding_model
        self.device = device
        self.type_to_idx = {'text': 0, 'visual': 1, 'hybrid': 2}
    
    def encode_query(self, query: str) -> np.ndarray:
        inputs = self.tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.squeeze().cpu().numpy().astype(np.float32)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        q_type, question = self.data[idx]
        embedding = self.encode_query(question)
        label = self.type_to_idx[q_type]
        return {
            'embedding': torch.tensor(embedding),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Load balanced data
print("\n[1/4] Loading balanced training data...")
data_path = Path('./data/balanced_training_data_v2.json')

if not data_path.exists():
    print(f"   ❌ Data not found at {data_path}")
    sys.exit(1)

with open(data_path, 'r') as f:
    data = json.load(f)

train_data = data['train']
val_data = data['val']
test_data = data['test']

print(f"   Train: {len(train_data)} samples")
print(f"   Val: {len(val_data)} samples")
print(f"   Test: {len(test_data)} samples")

# Print distribution
print(f"\n   Train distribution:")
for t in ['text', 'visual', 'hybrid']:
    count = sum(1 for q_type, _ in train_data if q_type == t)
    print(f"      {t}: {count} ({count/len(train_data)*100:.1f}%)")

# Load transformer
print("\n[2/4] Loading transformer model...")
device = torch.device(CONFIG['device'])
local_model_path = Path("./models/all-mpnet-base-v2")

if local_model_path.exists():
    tokenizer = AutoTokenizer.from_pretrained(str(local_model_path))
    embedding_model = AutoModel.from_pretrained(str(local_model_path))
    print(f"   ✅ Loaded from local")
else:
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    embedding_model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    print(f"   ✅ Loaded from cache")

embedding_model.to(device)
embedding_model.eval()

# Create datasets
print("\n[3/4] Creating datasets...")
train_dataset = BalancedDataset(train_data, tokenizer, embedding_model, device)
val_dataset = BalancedDataset(val_data, tokenizer, embedding_model, device)

train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

print(f"   Train batches: {len(train_loader)}")
print(f"   Val batches: {len(val_loader)}")

# Initialize model
print("\n[4/4] Training...")
model = BalancedRouterModel(
    CONFIG['embedding_dim'],
    CONFIG['hidden_dim'],
    CONFIG['num_classes'],
    CONFIG['dropout']
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

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
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'config': CONFIG
        }, './checkpoints/balanced_router_v2.pth')
        print(f"   ✅ Saved best model (val_acc: {val_acc*100:.2f}%)")

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print(f"   Best model: ./checkpoints/balanced_router_v2.pth")
print(f"   Best validation accuracy: {best_val_acc*100:.2f}%")
print("="*70)