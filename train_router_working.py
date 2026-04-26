"""
Working training script that matches your model's actual output keys
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Optional
from tqdm import tqdm
import argparse
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from cost_aware_visual_router import (
    UtilityPredictionModel,
)

class RouterDataset(Dataset):
    """Dataset for router training"""
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.samples = self._load_samples()

    def _load_samples(self) -> List[Dict]:
        samples = []
        for file_path in sorted(self.data_path.glob("*.json")):
            try:
                with open(file_path, 'r') as f:
                    sample = json.load(f)
                
                required_keys = ['query_embedding', 'optimal_path', 'path_utilities', 'path_costs', 'quality_scores']
                if all(k in sample for k in required_keys):
                    if len(sample['query_embedding']) == 768 and len(sample['path_utilities']) == 4:
                        samples.append(sample)
            except Exception as e:
                continue
        print(f"Loaded {len(samples)} valid samples")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        return {
            'query_embedding': torch.tensor(sample['query_embedding'], dtype=torch.float32),
            'optimal_path': torch.tensor(sample['optimal_path'], dtype=torch.long),
            'path_utilities': torch.tensor(sample['path_utilities'], dtype=torch.float32),
            'path_costs': torch.tensor(sample['path_costs'], dtype=torch.float32),
            'quality_scores': torch.tensor(sample['quality_scores'], dtype=torch.float32),
        }


class RouterTrainer:
    """Trainer that matches your model's actual output keys"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        
    def train_epoch(self, train_loader, optimizer, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        utility_loss_total = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            # Move to device
            embeddings = batch['query_embedding'].to(self.device)
            targets = batch['optimal_path'].to(self.device)
            target_utilities = batch['path_utilities'].to(self.device)
            
            # Forward pass
            outputs = self.model(embeddings)
            
            # Get logits (classification)
            logits = outputs['path_logits']
            
            # Classification loss
            loss = self.criterion(logits, targets)
            
            # Optional: Add utility loss if available
            if 'utilities' in outputs:
                utility_loss = nn.MSELoss()(outputs['utilities'], target_utilities)
                loss = loss + 0.1 * utility_loss  # Small weight for utility
                utility_loss_total += utility_loss.item()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return {'total': avg_loss, 'accuracy': accuracy}
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                embeddings = batch['query_embedding'].to(self.device)
                targets = batch['optimal_path'].to(self.device)
                
                outputs = self.model(embeddings)
                logits = outputs['path_logits']
                
                loss = self.criterion(logits, targets)
                
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == targets).sum().item()
                total += targets.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return {'total': avg_loss, 'accuracy': accuracy}
    
    def predict_with_confidence(self, embedding):
        """Get prediction with confidence score"""
        self.model.eval()
        with torch.no_grad():
            if not isinstance(embedding, torch.Tensor):
                embedding = torch.tensor(embedding).to(self.device)
            if embedding.dim() == 1:
                embedding = embedding.unsqueeze(0)
            
            outputs = self.model(embedding)
            logits = outputs['path_logits']
            probs = torch.softmax(logits, dim=1)
            confidence, prediction = torch.max(probs, dim=1)
            
            return prediction.item(), confidence.item(), outputs


def generate_balanced_training_data(num_samples=1000):
    """Generate balanced training data with clear path distinctions"""
    from sentence_transformers import SentenceTransformer
    
    output_dir = Path('./training_data_balanced')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading embedding model...")
    encoder = SentenceTransformer('all-mpnet-base-v2')
    
    # Define clear, distinct queries for each path
    path_queries = {
        0: [  # Technical/Programming
            "Python exception handling best practices",
            "JavaScript async await tutorial", 
            "React hooks useEffect dependency array",
            "Docker compose YAML configuration",
            "Git rebase vs merge conflict resolution",
            "SQL query optimization techniques",
            "REST API endpoint design patterns",
            "TypeScript interface vs type alias",
        ],
        1: [  # Medical/Healthcare
            "Myocardial infarction diagnosis criteria",
            "Type 2 diabetes insulin resistance mechanism",
            "COVID-19 spike protein antibody response",
            "Alzheimer's disease amyloid beta plaques",
            "Hypertension ACE inhibitor medication",
            "Asthma bronchodilator treatment protocol",
            "Stroke thrombolytic therapy window",
            "Breast cancer HER2 targeted therapy",
        ],
        2: [  # Legal
            "Fourth amendment unreasonable search seizure",
            "Title VII employment discrimination burden proof",
            "Copyright fair use transformative test",
            "Patent obviousness standard",
            "GDPR Article 17 right to erasure",
            "Contract frustration impossibility doctrine",
            "Tort negligence duty of care breach",
            "Criminal mens rea intent requirement",
        ],
        3: [  # Business
            "SWOT analysis strategic planning framework",
            "KPI dashboard metrics for SaaS companies",
            "Customer lifetime value calculation formula",
            "A/B testing statistical significance calculator",
            "Supply chain just-in-time inventory management",
            "Customer acquisition cost reduction strategies",
            "Net promoter score survey methodology",
            "Price elasticity demand curve estimation",
        ]
    }
    
    samples_per_path = num_samples // 4
    sample_id = 0
    
    for path_id in range(4):
        print(f"\nGenerating samples for Path {path_id}...")
        queries = path_queries[path_id]
        
        for i in range(samples_per_path):
            # Cycle through queries
            base_query = queries[i % len(queries)]
            
            # Create variations
            variations = [
                base_query,
                f"Explain: {base_query}",
                f"How to understand {base_query}",
                f"Guide to {base_query}",
                f"Understanding {base_query}",
                f"Tell me about {base_query}",
            ]
            query = variations[i % len(variations)]
            
            # Get embedding
            embedding = encoder.encode(query).astype(np.float32)
            
            # Create strong utility differences (optimal path much better)
            utilities = np.random.uniform(0.1, 0.3, 4)
            utilities[path_id] = np.random.uniform(0.85, 0.98)
            
            # Different costs for different paths
            costs = np.array([3.0, 4.0, 5.0, 6.0])
            costs[path_id] = np.random.uniform(1.0, 2.0)
            
            # Quality scores
            quality = np.random.uniform(0.2, 0.4, 4)
            quality[path_id] = np.random.uniform(0.9, 0.99)
            
            sample = {
                'query': query,
                'query_embedding': embedding.tolist(),
                'optimal_path': path_id,
                'path_utilities': utilities.tolist(),
                'path_costs': costs.tolist(),
                'quality_scores': quality.tolist(),
            }
            
            # Save
            output_path = output_dir / f"sample_{sample_id:06d}.json"
            with open(output_path, 'w') as f:
                json.dump(sample, f, indent=2)
            
            sample_id += 1
            
            if (i + 1) % 100 == 0:
                print(f"  Generated {i+1}/{samples_per_path} samples")
    
    print(f"\n✅ Generated {sample_id} total samples in {output_dir}")
    
    # Show distribution
    print("\nSample distribution by path:")
    for path_id in range(4):
        count = sum(1 for f in output_dir.glob("*.json") 
                   if f.stat().st_size > 0) // 4
        print(f"  Path {path_id}: ~{count} samples")
    
    return sample_id

def train_router(config, train_loader, val_loader):
    """Training function with TensorBoard logging"""
    
    log_dir = f"./runs/router_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir)
    print(f"📊 TensorBoard logs will be saved to: {log_dir}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = UtilityPredictionModel(
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_paths=config['num_paths'],
        dropout=config['dropout']
    ).to(device)
    
    trainer = RouterTrainer(model, device=device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['num_epochs']
    )
    
    best_val_accuracy = 0
    best_model_path = None
    
    print("\nStarting training...\n")
    
    for epoch in range(1, config['num_epochs'] + 1):
        # Training
        train_metrics = trainer.train_epoch(train_loader, optimizer, epoch)
        
        # Validation
        val_metrics = trainer.validate(val_loader)
        
        # ============================================
        # ADD TENSORBOARD LOGGING HERE (After metrics are computed)
        # ============================================
        # Log training metrics to TensorBoard
        writer.add_scalar('Loss/train_total', train_metrics['total'], epoch)
        writer.add_scalar('Accuracy/train', train_metrics['accuracy'], epoch)
        
        # Log validation metrics
        writer.add_scalar('Loss/val_total', val_metrics['total'], epoch)
        writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch)
        
        # Log learning rate
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning_rate', current_lr, epoch)
        
        # Print progress (keep your existing print statements)
        print(f"\nEpoch {epoch}/{config['num_epochs']}")
        print(f"  Train Loss: {train_metrics['total']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        print(f"  Val Loss: {val_metrics['total']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
        
        scheduler.step()
        
        # Save best model based on accuracy
        if val_metrics['accuracy'] > best_val_accuracy:
            best_val_accuracy = val_metrics['accuracy']
            best_model_path = f"./checkpoints/best_router.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_accuracy': val_metrics['accuracy'],
                'config': config
            }, best_model_path)
            print(f"  → Saved best model (Accuracy: {best_val_accuracy:.4f})")
    
    # ============================================
    # CLOSE TENSORBOARD WRITER (After training loop, before return)
    # ============================================
    writer.close()
    print(f"\n✅ TensorBoard logs saved! View with: tensorboard --logdir runs")
    
    # Load best model
    if best_model_path:
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\nLoaded best model with accuracy: {checkpoint['val_accuracy']:.4f}")
    
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./training_data_balanced')
    parser.add_argument('--generate_data', action='store_true')
    parser.add_argument('--num_samples', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    
    # Generate data if requested
    if args.generate_data:
        print("Generating balanced training data...")
        generate_balanced_training_data(args.num_samples)
    
    # Load dataset
    print(f"\nLoading dataset from {args.data_dir}")
    dataset = RouterDataset(args.data_dir)
    
    if len(dataset) == 0:
        print("ERROR: No data found! Please generate data first with --generate_data")
        return
    
    # Split train/val
    val_size = int(len(dataset) * 0.1)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Train samples: {train_size}, Val samples: {val_size}")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    config = {
        'embedding_dim': 768,
        'hidden_dim': 256,
        'num_paths': 4,
        'dropout': 0.1,
        'num_epochs': args.num_epochs,
        'lr': args.lr,
        'weight_decay': 1e-5,
    }
    
    # Train model
    print("\n" + "="*60)
    print("TRAINING ROUTER")
    print("="*60)
    model = train_router(config, train_loader, val_loader)
    
    # Save final model
    Path('./checkpoints').mkdir(exist_ok=True)
    final_path = Path('./checkpoints/router_working_final.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config
    }, final_path)
    
    print(f"\n✅ Training complete! Model saved to {final_path}")
    
    # Quick test with different domains
    print("\n" + "="*60)
    print("TESTING WITH REAL QUERIES")
    print("="*60)
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer('all-mpnet-base-v2')
    
    test_queries = [
        ("Technical", "Python exception handling best practices"),
        ("Medical", "COVID-19 spike protein antibody response"),
        ("Legal", "GDPR Article 17 right to erasure"),
        ("Business", "Customer lifetime value calculation formula"),
        ("Mixed", "How to train a neural network"),  # Should go to Technical
        ("Mixed", "Marketing strategy for small business"),  # Should go to Business
    ]
    
    model.eval()
    print("\nPredictions:")
    print("-" * 50)
    with torch.no_grad():
        for domain, query in test_queries:
            embedding = encoder.encode(query).astype(np.float32)
            emb_tensor = torch.tensor(embedding).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')
            outputs = model(emb_tensor)
            probs = torch.softmax(outputs['path_logits'], dim=1)
            pred = torch.argmax(probs, dim=1).item()
            conf = probs[0, pred].item()
            
            # Get all probabilities
            probs_dict = {f"P{i}": probs[0, i].item() for i in range(4)}
            print(f"{domain:10} → Path {pred} (conf: {conf:.3f}) - {probs_dict}")


if __name__ == '__main__':
    main()