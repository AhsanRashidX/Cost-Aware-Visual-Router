"""
Train combined router using transformers (no sentence_transformers)
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
import random
import os

# Set environment variables
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

print("="*70)
print("TRAINING COMBINED ROUTER (Transformers Only)")
print("="*70)

# Configuration
CONFIG = {
    'embedding_dim': 768,
    'hidden_dim': 256,
    'num_classes': 3,  # text, visual, hybrid
    'dropout': 0.1,
    'batch_size': 64,
    'epochs': 30,
    'learning_rate': 1e-4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

print(f"\nConfiguration:")
for k, v in CONFIG.items():
    print(f"   {k}: {v}")

# Model definition
class CombinedRouterModel(nn.Module):
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

# Dataset class
class CombinedDataset(Dataset):
    def __init__(self, data, tokenizer, embedding_model, device):
        self.data = data
        self.tokenizer = tokenizer
        self.embedding_model = embedding_model
        self.device = device
        self.type_to_idx = {'text': 0, 'visual': 1, 'hybrid': 2}
    
    def __len__(self):
        return len(self.data)
    
    def encode_query(self, query: str) -> np.ndarray:
        """Encode query using transformer model"""
        inputs = self.tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
            # Mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings.squeeze().cpu().numpy().astype(np.float32)
    
    def __getitem__(self, idx):
        q_type, question = self.data[idx]
        embedding = self.encode_query(question)
        label = self.type_to_idx[q_type]
        return {
            'embedding': torch.tensor(embedding),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Load data
print("\n[1/5] Loading combined training data...")
data_path = Path('./data/combined_training_data.json')

if not data_path.exists():
    print(f"   ❌ Data not found at {data_path}")
    print("   Run: python scripts/combined_data_loader.py first")
    sys.exit(1)

with open(data_path, 'r') as f:
    data = json.load(f)

train_data = data['train']
val_data = data['val']
test_data = data['test']

print(f"   Train: {len(train_data)} samples")
print(f"   Val: {len(val_data)} samples")
print(f"   Test: {len(test_data)} samples")

# Load transformer model
print("\n[2/5] Loading transformer model...")
local_model_path = Path("./models/all-mpnet-base-v2")
device = torch.device(CONFIG['device'])

if local_model_path.exists():
    tokenizer = AutoTokenizer.from_pretrained(str(local_model_path))
    embedding_model = AutoModel.from_pretrained(str(local_model_path))
    print(f"   ✅ Loaded from local: {local_model_path}")
else:
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    embedding_model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    print(f"   ✅ Loaded from cache")

embedding_model.to(device)
embedding_model.eval()

# Create datasets
print("\n[3/5] Creating datasets...")
train_dataset = CombinedDataset(train_data, tokenizer, embedding_model, device)
val_dataset = CombinedDataset(val_data, tokenizer, embedding_model, device)
test_dataset = CombinedDataset(test_data, tokenizer, embedding_model, device)

train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)

print(f"   Train batches: {len(train_loader)}")
print(f"   Val batches: {len(val_loader)}")

# Initialize model
print("\n[4/5] Initializing model...")
model = CombinedRouterModel(
    CONFIG['embedding_dim'],
    CONFIG['hidden_dim'],
    CONFIG['num_classes'],
    CONFIG['dropout']
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

print(f"   Device: {device}")
print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Training loop
print("\n[5/5] Training...")
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
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'config': CONFIG
        }, './checkpoints/combined_router_final.pth')
        print(f"   ✅ Saved best model (val_acc: {val_acc*100:.2f}%)")

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print(f"   Best model: ./checkpoints/combined_router_final.pth")
print(f"   Best validation accuracy: {best_val_acc*100:.2f}%")
print("="*70)

# Quick test
print("\nTesting model on sample queries...")
model.eval()
test_queries = [
    ("text", "What is Python programming?"),
    ("visual", "What does a neural network diagram look like?"),
    ("hybrid", "Explain how a car engine works with diagram"),
]

for q_type, query in test_queries:
    embedding = train_dataset.encode_query(query)
    emb_tensor = torch.tensor(embedding).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(emb_tensor)
        probs = torch.softmax(outputs['path_logits'], dim=1)
        pred = torch.argmax(probs, dim=1).item()
        pred_type = ['text', 'visual', 'hybrid'][pred]
        confidence = probs[0, pred].item()
    status = "✅" if pred_type == q_type else "❌"
    print(f"   {status} [{q_type}] → [{pred_type}] (conf: {confidence:.3f}): {query[:50]}...")