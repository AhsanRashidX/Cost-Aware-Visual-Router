"""
Evaluate the balanced router on test set with real cost tracking
"""

import sys
import os
from pathlib import Path

# Add project root to Python path (must come before other imports)
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, confusion_matrix
import time

# Now import the cost tracker from scripts folder
from cost_tracker import RealCostTracker

# Initialize cost tracker
cost_tracker = RealCostTracker()

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

print("="*70)
print("EVALUATING BALANCED ROUTER")
print("="*70)

# Load model
print("\n[1/4] Loading balanced router...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint_path = Path('./checkpoints/balanced_router_v2.pth')

if not checkpoint_path.exists():
    print(f"   ❌ Model not found at {checkpoint_path}")
    sys.exit(1)

checkpoint = torch.load(checkpoint_path, map_location=device)

from train_balanced_router_v2 import BalancedRouterModel
model = BalancedRouterModel(768, 256, 3, 0.1)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

print(f"   ✅ Model loaded - Val Acc: {checkpoint['val_acc']*100:.2f}%")
print(f"   Epoch: {checkpoint['epoch']+1}")

# Load transformer
print("\n[2/4] Loading transformer model...")
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

def encode_query(query: str) -> np.ndarray:
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = embedding_model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.squeeze().cpu().numpy().astype(np.float32)

# Load test data
print("\n[3/4] Loading test data...")
data_path = Path('./data/balanced_training_data_v2.json')
with open(data_path, 'r') as f:
    data = json.load(f)

test_data = data['test']
type_to_idx = data['type_mapping']
idx_to_type = {v: k for k, v in type_to_idx.items()}

print(f"   Test samples: {len(test_data)}")
print(f"   Distribution:")
for t in ['text', 'visual', 'hybrid']:
    count = sum(1 for q_type, _ in test_data if q_type == t)
    print(f"      {t}: {count} ({count/len(test_data)*100:.1f}%)")

# Evaluate
print("\n[4/4] Running evaluation...")

predictions = []
ground_truth = []
confidences = []
latencies = []
real_costs = []
for i, (q_type, question) in enumerate(test_data):
    start_time = time.time()
    
    embedding = encode_query(question)
    emb_tensor = torch.tensor(embedding).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(emb_tensor)
        probs = F.softmax(outputs['path_logits'], dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred].item()
    
    latency = (time.time() - start_time) * 1000
    
    # Determine path used
    if pred == 0:
        path_used = 'text'
    elif pred == 1:
        path_used = 'visual'
    else:
        path_used = 'hybrid'
    
    # ========== ADD REAL COST TRACKING HERE ==========
    # Log real cost for this query
    real_cost = cost_tracker.log_query_cost(
        query=question,
        path_used=path_used,
        actual_latency_ms=latency,
        gpu_time_seconds=latency/1000 if path_used in ['visual', 'hybrid'] else None
    )
    real_costs.append(real_cost['total_cost'])
    # =================================================
    
    
    predictions.append(pred)
    ground_truth.append(type_to_idx[q_type])
    confidences.append(confidence)
    latencies.append(latency)
    
    if (i + 1) % 50 == 0:
        print(f"   Processed: {i+1}/{len(test_data)}")

# ========== SAVE COST SESSION AT THE END ==========
cost_tracker.save_session()
# ==================================================

# Calculate metrics
accuracy = accuracy_score(ground_truth, predictions)
cm = confusion_matrix(ground_truth, predictions)

# Cost calculation
costs = []
for p in predictions:
    if p == 0:  # text
        costs.append(0.0003)
    elif p == 1:  # visual
        costs.append(0.01)
    else:  # hybrid
        costs.append(0.0105)

total_cost = sum(costs)
baseline_cost = len(test_data) * 0.01
cost_savings = ((baseline_cost - total_cost) / baseline_cost) * 100

print("\n" + "="*70)
print("BALANCED ROUTER RESULTS")
print("="*70)

print(f"\n📊 Overall Performance:")
print(f"   Test Accuracy: {accuracy*100:.2f}%")
print(f"   Avg Confidence: {np.mean(confidences):.3f}")
print(f"   Avg Latency: {np.mean(latencies):.1f}ms")
print(f"   Total Cost: ${total_cost:.4f}")
print(f"   Baseline Cost: ${baseline_cost:.4f}")
print(f"   Cost Savings: {cost_savings:.1f}%")

print(f"\n📊 Per-Class Accuracy:")
classes = ['text', 'visual', 'hybrid']
for i, cls in enumerate(classes):
    mask = [j for j, t in enumerate(ground_truth) if t == i]
    if mask:
        correct = sum(1 for j in mask if predictions[j] == i)
        acc = correct / len(mask) * 100
        status = "✅" if acc >= 85 else "⚠️" if acc >= 70 else "❌"
        print(f"   {status} {cls.upper()}: {acc:.1f}% ({correct}/{len(mask)})")

print(f"\n📊 Confusion Matrix:")
print(f"{'':12} {'Pred Text':10} {'Pred Visual':10} {'Pred Hybrid':10}")
print("-" * 45)
for i, true_class in enumerate(classes):
    print(f"{true_class.upper():10} {cm[i][0]:10} {cm[i][1]:10} {cm[i][2]:10}")

# Path usage
path_usage = {0: 0, 1: 0, 2: 0}
for p in predictions:
    path_usage[p] += 1

print(f"\n📊 Path Usage:")
print(f"   Text path: {path_usage[0]} ({path_usage[0]/len(predictions)*100:.1f}%)")
print(f"   Visual path: {path_usage[1]} ({path_usage[1]/len(predictions)*100:.1f}%)")
print(f"   Hybrid path: {path_usage[2]} ({path_usage[2]/len(predictions)*100:.1f}%)")

# Save results
results = {
    'test_accuracy': accuracy,
    'confusion_matrix': cm.tolist(),
    'per_class_accuracy': {
        classes[i]: {
            'accuracy': float(np.mean([predictions[j] == i for j, t in enumerate(ground_truth) if t == i])) if any(t == i for t in ground_truth) else 0,
            'correct': sum(1 for j, t in enumerate(ground_truth) if t == i and predictions[j] == i),
            'total': sum(1 for t in ground_truth if t == i)
        } for i in range(3)
    },
    'avg_confidence': float(np.mean(confidences)),
    'avg_latency_ms': float(np.mean(latencies)),
    'cost_savings_percent': cost_savings,
    'total_cost': total_cost,
    'baseline_cost': baseline_cost,
    'path_usage': path_usage
}

output_path = Path('./logs/balanced_router_results.json')
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n💾 Results saved to: {output_path}")

print("\n" + "="*70)
print("✅ Evaluation complete!")
print("="*70)