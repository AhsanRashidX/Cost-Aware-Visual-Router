"""
Test script to verify DocVQA data loading without router
"""

import json
from pathlib import Path

print("="*60)
print("TESTING DOCVQA DATA LOADING")
print("="*60)

DOCVQA_PATH = Path("./data/docvqa")

# Check if files exist
print("\n1. Checking file locations:")
annotations_dir = DOCVQA_PATH / "annotations"
print(f"   Annotations directory: {annotations_dir}")
print(f"   Exists: {annotations_dir.exists()}")

if annotations_dir.exists():
    files = list(annotations_dir.glob("*.json"))
    print(f"   JSON files found: {[f.name for f in files]}")

# Try loading each annotation file
print("\n2. Loading annotation files:")

def test_load_file(filepath, max_items=10):
    """Test loading a JSON file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"   ✅ {filepath.name}:")
        print(f"      Type: {type(data)}")
        
        if isinstance(data, dict):
            print(f"      Keys: {list(data.keys())[:5]}")
            if 'data' in data:
                items = data['data']
                print(f"      Items count: {len(items)}")
                # Show first few questions
                for i, item in enumerate(items[:max_items]):
                    question = item.get('question', item.get('questionText', 'N/A'))
                    print(f"        {i+1}. {question[:80]}...")
            elif 'annotations' in data:
                items = data['annotations']
                print(f"      Items count: {len(items)}")
        elif isinstance(data, list):
            print(f"      Items count: {len(data)}")
            for i, item in enumerate(data[:max_items]):
                question = item.get('question', item.get('questionText', 'N/A'))
                print(f"        {i+1}. {question[:80]}...")
        
        return True
    except Exception as e:
        print(f"   ❌ {filepath.name}: {e}")
        return False

# Test val.json
val_file = annotations_dir / "val.json"
if val_file.exists():
    test_load_file(val_file)
else:
    print(f"   ❌ val.json not found at {val_file}")

# Test test.json
test_file = annotations_dir / "test.json"
if test_file.exists():
    test_load_file(test_file)
else:
    print(f"   ❌ test.json not found at {test_file}")

# Test train.json
train_file = annotations_dir / "train.json"
if train_file.exists():
    test_load_file(train_file, max_items=5)
else:
    print(f"   ❌ train.json not found at {train_file}")

print("\n" + "="*60)