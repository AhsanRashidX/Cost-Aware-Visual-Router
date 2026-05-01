"""
Simple evaluation using only basic Python (no heavy imports)
"""

import json
import time
import random
from pathlib import Path

print("="*70)
print("SIMPLE EVALUATION (No Heavy Imports)")
print("="*70)

# Step 1: Load DocVQA data
print("\n[1/3] Loading DocVQA data...")

DOCVQA_PATH = Path("./data/docvqa")
annotations_dir = DOCVQA_PATH / "annotations"

def load_queries_from_file(filepath, max_queries=200):
    """Load queries from annotation file"""
    queries = []
    
    if not filepath.exists():
        print(f"   File not found: {filepath}")
        return queries
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, dict):
            if 'data' in data:
                items = data['data']
            elif 'annotations' in data:
                items = data['annotations']
            else:
                items = [data]
        elif isinstance(data, list):
            items = data
        else:
            return queries
        
        for item in items[:max_queries]:
            # Get question
            question = item.get('question', '')
            if not question:
                question = item.get('questionText', '')
            if not question:
                continue
            
            # Simple classification
            q_lower = question.lower()
            
            # Text queries (factual extraction)
            if any(kw in q_lower for kw in ['what is', 'who', 'when', 'where', 'how many']):
                q_type = 'text'
            # Visual queries (appearance-based)
            elif any(kw in q_lower for kw in ['look like', 'color', 'shape', 'diagram', 'show']):
                q_type = 'visual'
            # Hybrid (need both)
            else:
                q_type = 'hybrid'
            
            queries.append((q_type, question))
        
        print(f"   Loaded {len(queries)} queries from {filepath.name}")
        
    except Exception as e:
        print(f"   Error loading {filepath.name}: {e}")
    
    return queries

# Load all queries
all_queries = []

val_file = annotations_dir / "val.json"
if val_file.exists():
    queries = load_queries_from_file(val_file, max_queries=250)
    all_queries.extend(queries)

test_file = annotations_dir / "test.json"
if test_file.exists():
    queries = load_queries_from_file(test_file, max_queries=250)
    all_queries.extend(queries)

# If no real queries, use synthetic
if not all_queries:
    print("\n   No real queries found! Using synthetic queries...")
    synthetic = [
        ("text", "What is Python programming?"),
        ("text", "Explain machine learning"),
        ("text", "What is cloud computing?"),
        ("visual", "What does a neural network diagram look like?"),
        ("visual", "Show me DNA structure"),
        ("visual", "What does the Eiffel Tower look like?"),
        ("hybrid", "Explain how a car engine works with diagram"),
        ("hybrid", "How does photosynthesis work with illustration"),
    ]
    for _ in range(50):
        for q in synthetic:
            all_queries.append(q)

print(f"\n   Total queries: {len(all_queries)}")
text_count = sum(1 for t, _ in all_queries if t == 'text')
visual_count = sum(1 for t, _ in all_queries if t == 'visual')
hybrid_count = sum(1 for t, _ in all_queries if t == 'hybrid')
print(f"   Text: {text_count}, Visual: {visual_count}, Hybrid: {hybrid_count}")

# Step 2: Simple rule-based router (no ML model)
print("\n[2/3] Running simple rule-based router...")

def simple_router(query):
    """Rule-based router for baseline comparison"""
    q_lower = query.lower()
    
    # Check for hybrid patterns
    if ('explain' in q_lower or 'how does' in q_lower) and ('diagram' in q_lower or 'illustration' in q_lower):
        return 'hybrid', 0.85, 0.0105
    
    # Check for visual patterns
    visual_keywords = ['look like', 'show me', 'diagram', 'picture', 'image', 'what does']
    if any(kw in q_lower for kw in visual_keywords):
        return 'visual', 0.8, 0.01
    
    # Check for text patterns
    text_keywords = ['what is', 'who', 'when', 'where', 'define', 'explain']
    if any(kw in q_lower for kw in text_keywords):
        return 'text', 0.9, 0.0005
    
    # Default
    return 'text', 0.6, 0.0005

# Evaluate
results = []
correct = 0
total_cost = 0
total_latency = 0

for i, (expected_type, query) in enumerate(all_queries):
    start = time.time()
    predicted, confidence, cost = simple_router(query)
    latency = (time.time() - start) * 1000
    
    is_correct = (predicted == expected_type)
    if is_correct:
        correct += 1
    
    total_cost += cost
    total_latency += latency
    
    results.append({
        'query': query[:80],
        'expected': expected_type,
        'predicted': predicted,
        'correct': is_correct,
        'confidence': confidence,
        'cost': cost,
        'latency_ms': latency
    })
    
    if (i + 1) % 100 == 0:
        print(f"   Processed: {i+1}/{len(all_queries)}")

# Step 3: Calculate metrics
print("\n[3/3] Calculating results...")

total = len(results)
accuracy = correct / total
baseline_cost = total * 0.01
cost_savings = ((baseline_cost - total_cost) / baseline_cost) * 100

# Per-type accuracy
per_type = {}
for dtype in ['text', 'visual', 'hybrid']:
    type_results = [r for r in results if r['expected'] == dtype]
    if type_results:
        type_correct = sum(1 for r in type_results if r['correct'])
        per_type[dtype] = type_correct / len(type_results)
    else:
        per_type[dtype] = 0

# Print results
print("\n" + "="*70)
print("EVALUATION RESULTS (Rule-Based Router)")
print("="*70)

print(f"\n📊 Overall Performance:")
print(f"   Total Queries: {total}")
print(f"   Accuracy: {accuracy*100:.2f}%")
print(f"   Cost Savings: {cost_savings:.2f}%")
print(f"   Total Cost: ${total_cost:.4f}")
print(f"   Baseline Cost: ${baseline_cost:.4f}")
print(f"   Avg Latency: {total_latency/total:.1f}ms")

print(f"\n📊 Per-Type Accuracy:")
print(f"   Text: {per_type.get('text', 0)*100:.1f}%")
print(f"   Visual: {per_type.get('visual', 0)*100:.1f}%")
print(f"   Hybrid: {per_type.get('hybrid', 0)*100:.1f}%")

# Confusion matrix
print(f"\n📊 Confusion Matrix:")
cm = {'text': {'text': 0, 'visual': 0, 'hybrid': 0},
      'visual': {'text': 0, 'visual': 0, 'hybrid': 0},
      'hybrid': {'text': 0, 'visual': 0, 'hybrid': 0}}

for r in results:
    cm[r['expected']][r['predicted']] += 1

print(f"               Predicted")
print(f"              Text  Visual  Hybrid")
print(f"   Text       {cm['text']['text']:4d}   {cm['text']['visual']:4d}    {cm['text']['hybrid']:4d}")
print(f"   Visual      {cm['visual']['text']:4d}   {cm['visual']['visual']:4d}    {cm['visual']['hybrid']:4d}")
print(f"   Hybrid      {cm['hybrid']['text']:4d}   {cm['hybrid']['visual']:4d}    {cm['hybrid']['hybrid']:4d}")

# Save results
output = {
    'total_queries': total,
    'accuracy': accuracy,
    'cost_savings_percent': cost_savings,
    'total_cost': total_cost,
    'baseline_cost': baseline_cost,
    'per_type_accuracy': per_type,
    'confusion_matrix': {k: v for k, v in cm.items()},
    'results': results
}

output_path = Path('./logs/results/simple_evaluation.json')
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n💾 Results saved to: {output_path}")

print("\n" + "="*70)
print("✅ Evaluation complete!")
print("="*70)