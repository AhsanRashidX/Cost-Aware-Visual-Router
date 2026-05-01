# scripts/create_training_data.py
"""
Create training data from DocVQA for router retraining
"""

import json
from pathlib import Path
from collections import defaultdict

DOCVQA_PATH = Path("./data/docvqa")
annotations_dir = DOCVQA_PATH / "annotations"

def classify_for_training(question):
    """Classification based on DocVQA question patterns"""
    q_lower = question.lower()
    
    # Visual questions (need to see location/layout)
    if 'where' in q_lower or 'located' in q_lower:
        return 'visual'
    
    # Hybrid questions (why/how)
    if q_lower.startswith(('why', 'how')):
        return 'hybrid'
    
    # Text questions (extraction)
    return 'text'

# Load and create training data
train_queries = []
val_queries = []

for split in ['train', 'val']:
    ann_file = annotations_dir / f"{split}.json"
    with open(ann_file, 'r') as f:
        data = json.load(f)
    
    items = data.get('data', [])
    for item in items[:1000]:  # 1000 samples per split
        question = item.get('question', '')
        if question:
            q_type = classify_for_training(question)
            if split == 'train':
                train_queries.append((q_type, question))
            else:
                val_queries.append((q_type, question))

print(f"Training samples: {len(train_queries)}")
print(f"  Text: {sum(1 for t,_ in train_queries if t=='text')}")
print(f"  Visual: {sum(1 for t,_ in train_queries if t=='visual')}")
print(f"  Hybrid: {sum(1 for t,_ in train_queries if t=='hybrid')}")

# Save for training
output = {'train': train_queries, 'val': val_queries}
with open('./data/docvqa_router_data.json', 'w') as f:
    json.dump(output, f, indent=2)