"""
Evaluate fixed router on DocVQA dataset
"""

import sys
import json
import time
import re
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

print("="*70)
print("FIXED ROUTER EVALUATION - DocVQA")
print("="*70)

# Step 1: Load fixed router
print("\n[1/5] Loading fixed router...")

try:
    from scripts.fixed_router import FixedRouter
    router = FixedRouter()
except Exception as e:
    print(f"   ❌ Failed to load router: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 2: Load DocVQA data
print("\n[2/5] Loading DocVQA queries...")

DOCVQA_PATH = Path("./data/docvqa")
annotations_dir = DOCVQA_PATH / "annotations"

def classify_ground_truth(question):
    """Ground truth classification for DocVQA"""
    q_lower = question.lower()
    
    # Hybrid: 'why' or 'how' questions
    if q_lower.startswith(('why', 'how')):
        return 'hybrid'
    
    # Visual: 'where' questions about location
    if 'where' in q_lower or 'located' in q_lower:
        return 'visual'
    
    # Default: text extraction
    return 'text'

def load_queries(filepath, max_queries=300):
    """Load queries from annotation file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    queries = []
    items = data.get('data', [])
    
    for item in items[:max_queries]:
        question = item.get('question', '')
        if question:
            q_type = classify_ground_truth(question)
            queries.append((q_type, question))
    
    return queries

# Load queries
val_queries = load_queries(annotations_dir / "val.json", max_queries=300)
test_queries = load_queries(annotations_dir / "test.json", max_queries=300)
all_queries = val_queries + test_queries

# Print distribution
type_counts = defaultdict(int)
for t, _ in all_queries:
    type_counts[t] += 1

print(f"\n   Total queries: {len(all_queries)}")
print(f"   Text: {type_counts.get('text', 0)} ({type_counts.get('text', 0)/len(all_queries)*100:.1f}%)")
print(f"   Visual: {type_counts.get('visual', 0)} ({type_counts.get('visual', 0)/len(all_queries)*100:.1f}%)")
print(f"   Hybrid: {type_counts.get('hybrid', 0)} ({type_counts.get('hybrid', 0)/len(all_queries)*100:.1f}%)")

# Step 3: Run evaluation
print("\n[3/5] Running evaluation...")

def get_path_category(path_name):
    if 'Parametric' in path_name or 'Text-Only' in path_name:
        return 'text'
    elif 'Visual-Only' in path_name:
        return 'visual'
    elif 'Hybrid' in path_name:
        return 'hybrid'
    return 'unknown'

results = []
correct = 0
confidences = []
costs = []
latencies = []
confusion = {'text': {'text': 0, 'visual': 0, 'hybrid': 0},
             'visual': {'text': 0, 'visual': 0, 'hybrid': 0},
             'hybrid': {'text': 0, 'visual': 0, 'hybrid': 0}}
path_usage = {'text': 0, 'visual': 0, 'hybrid': 0}

for i, (expected, query) in enumerate(all_queries):
    start = time.time()
    
    try:
        result = router.route_and_retrieve(query)
        latency = (time.time() - start) * 1000
        
        path_name = result.get('path_name', 'unknown')
        predicted = get_path_category(path_name)
        confidence = result.get('confidence', 0.5)
        cost = result.get('retrieval_result', {}).get('cost', 0.01)
        
        is_correct = (predicted == expected)
        if is_correct:
            correct += 1
        
        confusion[expected][predicted] += 1
        path_usage[predicted] += 1
        confidences.append(confidence)
        costs.append(cost)
        latencies.append(latency)
        
        results.append({
            'query': query[:80],
            'expected': expected,
            'predicted': predicted,
            'confidence': confidence,
            'correct': is_correct
        })
        
    except Exception as e:
        print(f"   Error on query {i}: {e}")
        results.append({
            'query': query[:80],
            'expected': expected,
            'predicted': 'ERROR',
            'correct': False
        })
    
    if (i + 1) % 100 == 0:
        print(f"   Processed: {i+1}/{len(all_queries)}")

# Step 4: Calculate metrics
print("\n[4/5] Calculating metrics...")

total = len(results)
accuracy = correct / total if total > 0 else 0
total_cost = sum(costs)
baseline_cost = total * 0.01
cost_savings = ((baseline_cost - total_cost) / baseline_cost) * 100 if baseline_cost > 0 else 0

# Per-type accuracy
per_type_acc = {}
for dtype in ['text', 'visual', 'hybrid']:
    type_total = sum(1 for r in results if r['expected'] == dtype)
    type_correct = sum(1 for r in results if r['expected'] == dtype and r.get('correct', False))
    per_type_acc[dtype] = type_correct / type_total if type_total > 0 else 0

# Step 5: Print results
print("\n[5/5] Results:")

print("\n" + "="*70)
print("FIXED ROUTER RESULTS")
print("="*70)

print(f"\n📊 Overall Performance:")
print(f"   Total Queries: {total}")
print(f"   Accuracy: {accuracy*100:.2f}%")
print(f"   Cost Savings: {cost_savings:.2f}%")
print(f"   Total Cost: ${total_cost:.4f}")
print(f"   Baseline Cost: ${baseline_cost:.4f}")
print(f"   Avg Confidence: {sum(confidences)/len(confidences):.3f}" if confidences else "   Avg Confidence: N/A")
print(f"   Avg Latency: {sum(latencies)/len(latencies):.1f}ms" if latencies else "   Avg Latency: N/A")

print(f"\n📊 Path Usage:")
print(f"   Text path: {path_usage['text']} ({path_usage['text']/total*100:.1f}%)")
print(f"   Visual path: {path_usage['visual']} ({path_usage['visual']/total*100:.1f}%)")
print(f"   Hybrid path: {path_usage['hybrid']} ({path_usage['hybrid']/total*100:.1f}%)")

print(f"\n📊 Per-Type Accuracy:")
for dtype in ['text', 'visual', 'hybrid']:
    status = "✅" if per_type_acc[dtype] >= 0.85 else "⚠️" if per_type_acc[dtype] >= 0.70 else "❌"
    print(f"   {status} {dtype.upper()}: {per_type_acc[dtype]*100:.1f}%")

print(f"\n📊 Confusion Matrix:")
print(f"{'':12} {'Pred Text':10} {'Pred Visual':10} {'Pred Hybrid':10}")
print("-" * 45)
for exp in ['text', 'visual', 'hybrid']:
    print(f"{exp.upper():10} {confusion[exp]['text']:10} {confusion[exp]['visual']:10} {confusion[exp]['hybrid']:10}")

# Save results
output = {
    'router_type': 'fixed_trained_router',
    'total_queries': total,
    'accuracy': accuracy,
    'cost_savings_percent': cost_savings,
    'total_cost': total_cost,
    'baseline_cost': baseline_cost,
    'avg_confidence': sum(confidences)/len(confidences) if confidences else 0,
    'avg_latency_ms': sum(latencies)/len(latencies) if latencies else 0,
    'per_type_accuracy': per_type_acc,
    'path_usage': path_usage,
    'confusion_matrix': confusion,
    'type_distribution': dict(type_counts)
}

output_path = Path('./logs/results/fixed_router_docvqa.json')
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

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

print("\n" + "="*70)