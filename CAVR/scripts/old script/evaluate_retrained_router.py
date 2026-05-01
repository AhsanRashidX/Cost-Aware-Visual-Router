"""
Evaluate the correctly matched router model
"""

import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time
import os

# Set environment variables
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

print("="*70)
print("EVALUATING CORRECT ROUTER MODEL")
print("="*70)

# Load model
print("\n[1/6] Loading trained router...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"   Device: {device}")

checkpoint_path = Path('./checkpoints/docvqa_router_fixed.pth')
if not checkpoint_path.exists():
    print(f"   ❌ Checkpoint not found at {checkpoint_path}")
    sys.exit(1)

print(f"   Loading from: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location=device)

# Import the correct model
from correct_router_model import CorrectRouterModel

model = CorrectRouterModel(input_dim=768, hidden_dim=256, num_classes=4, dropout=0.1)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

print(f"   ✅ Model loaded successfully")
print(f"   Validation accuracy: {checkpoint['val_acc']*100:.2f}%")
print(f"   Epoch: {checkpoint['epoch']+1}")

# Load transformer for embeddings
print("\n[2/6] Loading transformer model for embeddings...")
local_model_path = Path("./models/all-mpnet-base-v2")

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

def encode_query(query: str) -> np.ndarray:
    """Encode query using transformer model"""
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = embedding_model(**inputs)
        # Mean pooling
        embeddings = outputs.last_hidden_state.mean(dim=1)
    
    return embeddings.squeeze().cpu().numpy().astype(np.float32)

# Load test data
print("\n[3/6] Loading test data...")
test_data_path = Path('./data/docvqa_training_data.json')
if not test_data_path.exists():
    print(f"   ❌ Test data not found at {test_data_path}")
    sys.exit(1)

with open(test_data_path, 'r') as f:
    data = json.load(f)

test_data = data['test']
type_to_idx = data['type_mapping']
idx_to_type = {v: k for k, v in type_to_idx.items()}

print(f"   Test samples: {len(test_data)}")
print(f"   Class mapping: {type_to_idx}")

# Run evaluation
print("\n[4/6] Running evaluation...")

predictions = []
ground_truth = []
confidences = []
costs = []
latencies = []

for i, (q_type, question) in enumerate(test_data):
    start_time = time.time()
    
    try:
        # Encode
        embedding = encode_query(question)
        emb_tensor = torch.tensor(embedding).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(emb_tensor)
            probs = F.softmax(outputs['path_logits'], dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred].item()
        
        latency = (time.time() - start_time) * 1000
        
        predictions.append(pred)
        ground_truth.append(type_to_idx[q_type])
        confidences.append(confidence)
        latencies.append(latency)
        
        # Estimate cost based on prediction
        if pred == 0:  # text -> parametric/text path
            cost = 0.0003
        elif pred == 1:  # visual
            cost = 0.01
        else:  # hybrid (class 2)
            cost = 0.0105
        costs.append(cost)
        
    except Exception as e:
        print(f"   Error on sample {i}: {e}")
        predictions.append(0)
        ground_truth.append(type_to_idx[q_type])
        confidences.append(0.5)
        latencies.append(0)
        costs.append(0.01)
    
    if (i + 1) % 200 == 0:
        print(f"   Processed: {i+1}/{len(test_data)}")

# Calculate metrics
print("\n[5/6] Calculating metrics...")

accuracy = accuracy_score(ground_truth, predictions)
cm = confusion_matrix(ground_truth, predictions)
total_cost = sum(costs)
baseline_cost = len(test_data) * 0.01
cost_savings = ((baseline_cost - total_cost) / baseline_cost) * 100

# Per-class metrics
classes = ['text', 'visual', 'hybrid']
per_class_acc = {}

for i, cls in enumerate(classes):
    mask = [j for j, t in enumerate(ground_truth) if t == i]
    if mask:
        correct = sum(1 for j in mask if predictions[j] == i)
        per_class_acc[cls] = correct / len(mask) * 100
    else:
        per_class_acc[cls] = 0.0

print("\n[6/6] Results:")

print("\n" + "="*70)
print("FINAL RESULTS - CORRECT ROUTER MODEL")
print("="*70)

print(f"\n📊 Overall Performance:")
print(f"   Test Accuracy: {accuracy*100:.2f}%")
print(f"   Avg Confidence: {np.mean(confidences):.3f}")
print(f"   Avg Latency: {np.mean(latencies):.1f}ms")
print(f"   Total Cost: ${total_cost:.4f}")
print(f"   Baseline Cost: ${baseline_cost:.4f}")
print(f"   Cost Savings: {cost_savings:.1f}%")

print(f"\n📊 Per-Class Accuracy:")
for cls in classes:
    acc = per_class_acc[cls]
    status = "✅" if acc >= 85 else "⚠️" if acc >= 70 else "❌"
    print(f"   {status} {cls.upper()}: {acc:.1f}%")

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
print(f"   Text path (parametric/text): {path_usage[0]} ({path_usage[0]/len(predictions)*100:.1f}%)")
print(f"   Visual path: {path_usage[1]} ({path_usage[1]/len(predictions)*100:.1f}%)")
print(f"   Hybrid path: {path_usage[2]} ({path_usage[2]/len(predictions)*100:.1f}%)")

# Save results
results = {
    'test_accuracy': accuracy,
    'confusion_matrix': cm.tolist(),
    'per_class_accuracy': per_class_acc,
    'avg_confidence': float(np.mean(confidences)),
    'avg_latency_ms': float(np.mean(latencies)),
    'cost_savings_percent': cost_savings,
    'total_cost': total_cost,
    'baseline_cost': baseline_cost,
    'path_usage': path_usage,
    'total_samples': len(test_data)
}

output_path = Path('./logs/correct_router_results.json')
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n💾 Results saved to: {output_path}")

# Target achievement
print("\n" + "="*70)
print("TARGET ACHIEVEMENT")
print("="*70)

if accuracy >= 0.85:
    print(f"   ✅ Accuracy target (≥85%): {accuracy*100:.1f}% ACHIEVED")
else:
    print(f"   ❌ Accuracy target (≥85%): {accuracy*100:.1f}% NOT ACHIEVED")

if cost_savings >= 40:
    print(f"   ✅ Cost savings target (≥40%): {cost_savings:.1f}% ACHIEVED")
else:
    print(f"   ❌ Cost savings target (≥40%): {cost_savings:.1f}% NOT ACHIEVED")

# Show sample predictions
print("\n" + "="*70)
print("SAMPLE PREDICTIONS")
print("="*70)

# Correct predictions
print("\n✅ Correct predictions:")
correct_count = 0
for (q_type, q), pred, conf in zip(test_data, predictions, confidences):
    if pred == type_to_idx[q_type] and correct_count < 5:
        pred_class = idx_to_type[pred]
        print(f"   [{q_type}] → [{pred_class}] (conf: {conf:.3f})")
        print(f"      {q[:70]}...")
        correct_count += 1

# Incorrect predictions
print("\n❌ Incorrect predictions:")
incorrect_count = 0
for (q_type, q), pred, conf in zip(test_data, predictions, confidences):
    if pred != type_to_idx[q_type] and incorrect_count < 10:
        pred_class = idx_to_type[pred]
        print(f"   [{q_type}] → [{pred_class}] (conf: {conf:.3f})")
        print(f"      {q[:70]}...")
        incorrect_count += 1

print("\n" + "="*70)
print("✅ Evaluation complete!")
print("="*70)