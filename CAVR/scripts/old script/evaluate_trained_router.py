"""
Evaluate trained router on DocVQA dataset
"""

import sys
import json
import time
import re
from pathlib import Path
from collections import defaultdict

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

print("="*70)
print("TRAINED ROUTER EVALUATION - DocVQA")
print("="*70)

# Step 1: Load trained router
print("\n[1/5] Loading trained router...")

try:
    from colpali_router_demo import CompleteVisualRouter
    router = CompleteVisualRouter()
    print("   ✅ Trained router loaded successfully")
except Exception as e:
    print(f"   ❌ Failed to load router: {e}")
    print("   Using lightweight wrapper...")
    
    # Fallback to lightweight router
    from scripts.lightweight_router import LightweightRouter
    router = LightweightRouter()
    print("   ✅ Lightweight router loaded")

# Step 2: Load DocVQA data with same classification
print("\n[2/5] Loading DocVQA queries...")

DOCVQA_PATH = Path("./data/docvqa")
annotations_dir = DOCVQA_PATH / "annotations"

def classify_docvqa_question(question: str) -> str:
    """Same classification as before"""
    q_lower = question.lower()
    
    # Text extraction questions
    text_patterns = [
        r'what is the (name|title|date|value|amount|number)',
        r'what (year|time|location|address)',
        r'who (is|are)',
        r'how many',
        r'to whom',
        r'according to',
        r'what (percentage|percent|%)',
    ]
    
    # Visual questions
    visual_patterns = [
        r'where (is|are)',
        r'what (color|shape|size|position)',
        r'look like',
        r'show',
        r'diagram',
        r'graph',
        r'chart',
        r'figure',
        r'table',
        r'layout',
        r'located',
    ]
    
    # Hybrid questions
    hybrid_patterns = [
        r'explain',
        r'how (does|do|is|are)',
        r'why',
        r'what does this mean',
        r'compare',
        r'contrast',
        r'difference between',
        r'trend',
        r'pattern',
        r'conclusion',
    ]
    
    for pattern in hybrid_patterns:
        if re.search(pattern, q_lower):
            return 'hybrid'
    
    for pattern in visual_patterns:
        if re.search(pattern, q_lower):
            return 'visual'
    
    for pattern in text_patterns:
        if re.search(pattern, q_lower):
            return 'text'
    
    return 'text'

def load_queries(filepath, max_queries=250):
    """Load queries from annotation file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    queries = []
    items = data.get('data', [])
    
    for item in items[:max_queries]:
        question = item.get('question', '')
        if not question:
            continue
        q_type = classify_docvqa_question(question)
        queries.append((q_type, question))
    
    return queries

# Load queries
val_queries = load_queries(annotations_dir / "val.json", max_queries=250)
test_queries = load_queries(annotations_dir / "test.json", max_queries=250)
all_queries = val_queries + test_queries

# Print distribution
from collections import Counter
type_counts = Counter([t for t, _ in all_queries])
print(f"\n   Total queries: {len(all_queries)}")
print(f"   Text: {type_counts.get('text', 0)} ({type_counts.get('text', 0)/len(all_queries)*100:.1f}%)")
print(f"   Visual: {type_counts.get('visual', 0)} ({type_counts.get('visual', 0)/len(all_queries)*100:.1f}%)")
print(f"   Hybrid: {type_counts.get('hybrid', 0)} ({type_counts.get('hybrid', 0)/len(all_queries)*100:.1f}%)")

# Step 3: Run router evaluation
print("\n[3/5] Running router evaluation...")

def get_path_category(path_name):
    """Convert path name to category"""
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
        # Use route_and_retrieve (more stable)
        result = router.route_and_retrieve(query)
        latency = (time.time() - start) * 1000
        
        path_name = result.get('path_name', 'unknown')
        predicted = get_path_category(path_name)
        confidence = result.get('confidence', 0.5)
        
        # Estimate cost
        if predicted == 'text':
            cost = 0.0003  # average of parametric and text
        elif predicted == 'visual':
            cost = 0.01
        else:
            cost = 0.0105
        
        is_correct = (predicted == expected)
        if is_correct:
            correct += 1
        
        confusion[expected][predicted] += 1
        path_usage[predicted] += 1
        confidences.append(confidence)
        costs.append(cost)
        latencies.append(latency)
        
        results.append({
            'query': query[:100],
            'expected': expected,
            'predicted': predicted,
            'path_name': path_name,
            'confidence': confidence,
            'correct': is_correct
        })
        
    except Exception as e:
        print(f"   Error on query {i}: {e}")
        confusion[expected]['text'] += 1  # fallback
        results.append({
            'query': query[:100],
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

# Print results
print("\n" + "="*70)
print("TRAINED ROUTER RESULTS")
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

# Step 5: Save results
print("\n[5/5] Saving results...")

output = {
    'router_type': 'trained',
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
    'type_distribution': dict(type_counts),
    'results': results
}

output_path = Path('./logs/results/trained_router_docvqa.json')
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

# Show sample misclassifications
print("\n❌ Sample Misclassifications (first 10):")
misclassified = [r for r in results if not r.get('correct', False)]
for i, r in enumerate(misclassified[:10]):
    print(f"\n   {i+1}. Expected: {r['expected']} → Predicted: {r.get('predicted', '?')}")
    print(f"      Query: {r['query'][:100]}...")