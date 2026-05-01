"""
Combined Data Loader for DocVQA + Real InfoVQA
Uses actual InfoVQA dataset for better visual/hybrid performance
"""

import json
import random
from pathlib import Path
from collections import defaultdict

def classify_question(question: str) -> str:
    """Classify question type based on content"""
    q_lower = question.lower()
    
    # Visual patterns (location, layout, appearance)
    visual_patterns = [
        'where', 'located', 'position', 'layout', 'color', 'shape',
        'look like', 'appear', 'what does the (diagram|chart|graph|figure|image) show',
        'what is (shown|depicted|illustrated)', 'arrangement', 'placement',
        'what color', 'what shape', 'how many (colors|shapes)'
    ]
    
    # Hybrid patterns (understanding, reasoning, relationships)
    hybrid_patterns = [
        'why', 'how (does|do|is|are|can|could)', 'explain', 'compare',
        'contrast', 'difference', 'relationship', 'trend', 'pattern',
        'what (causes|leads to|results in)', 'what can (you|we) (infer|conclude|learn)',
        'what information', 'what insight', 'what conclusion'
    ]
    
    # Check hybrid first
    for pattern in hybrid_patterns:
        if pattern in q_lower:
            return 'hybrid'
    
    # Check visual
    for pattern in visual_patterns:
        if pattern in q_lower:
            return 'visual'
    
    # Default to text
    return 'text'

def load_docvqa_data(data_path: Path, max_samples: int = None):
    """Load DocVQA data"""
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    train_data = []
    val_data = []
    test_data = []
    
    for split in ['train', 'val', 'test']:
        if split in data:
            items = data[split]
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, (list, tuple)) and len(item) == 2:
                        q_type, question = item
                    elif isinstance(item, dict):
                        q_type = item.get('type', classify_question(item.get('question', '')))
                        question = item.get('question', '')
                    else:
                        continue
                    
                    if split == 'train':
                        train_data.append((q_type, question))
                    elif split == 'val':
                        val_data.append((q_type, question))
                    else:
                        test_data.append((q_type, question))
    
    if max_samples and max_samples < len(train_data):
        train_data = train_data[:max_samples]
        if val_data:
            val_data = val_data[:max_samples // 5]
        if test_data:
            test_data = test_data[:max_samples // 5]
    
    return train_data, val_data, test_data

def load_infovqa_data(data_dir: Path, max_samples: int = None):
    """
    Load real InfoVQA data from annotations
    InfoVQA has more balanced distribution of visual/hybrid questions
    """
    annotations_dir = data_dir / 'annotations'
    
    train_data = []
    val_data = []
    test_data = []
    
    # Try different possible file names
    possible_files = {
        'train': ['train.json', 'train_v1.0.json', 'train_v1.0_withQT.json', 'training.json'],
        'val': ['val.json', 'val_v1.0.json', 'val_v1.0_withQT.json', 'validation.json'],
        'test': ['test.json', 'test_v1.0.json', 'test_v1.0_withQT.json']
    }
    
    for split, filenames in possible_files.items():
        for filename in filenames:
            filepath = annotations_dir / filename
            if filepath.exists():
                print(f"   Loading InfoVQA {split} from: {filename}")
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                items = data.get('data', []) if isinstance(data, dict) else data
                
                for item in items:
                    question = item.get('question', '')
                    if not question:
                        question = item.get('questionText', '')
                    if not question:
                        continue
                    
                    q_type = classify_question(question)
                    
                    if split == 'train':
                        train_data.append((q_type, question))
                    elif split == 'val':
                        val_data.append((q_type, question))
                    else:
                        test_data.append((q_type, question))
                
                break  # Found the file, move to next split
    
    # Print distribution for InfoVQA
    if train_data:
        print(f"\n   InfoVQA Distribution:")
        for name, data_split in [("Train", train_data), ("Val", val_data), ("Test", test_data)]:
            if data_split:
                counts = defaultdict(int)
                for t, _ in data_split:
                    counts[t] += 1
                total = len(data_split)
                print(f"      {name}: {total} samples")
                print(f"         Text: {counts.get('text', 0)} ({counts.get('text', 0)/total*100:.1f}%)")
                print(f"         Visual: {counts.get('visual', 0)} ({counts.get('visual', 0)/total*100:.1f}%)")
                print(f"         Hybrid: {counts.get('hybrid', 0)} ({counts.get('hybrid', 0)/total*100:.1f}%)")
    
    if max_samples and max_samples < len(train_data):
        train_data = train_data[:max_samples]
        if val_data:
            val_data = val_data[:max_samples // 5]
        if test_data:
            test_data = test_data[:max_samples // 5]
    
    return train_data, val_data, test_data

def prepare_combined_data():
    """Prepare combined training data from DocVQA and InfoVQA"""
    
    print("="*60)
    print("Preparing Combined Training Data")
    print("(DocVQA + Real InfoVQA)")
    print("="*60)
    
    all_train = []
    all_val = []
    all_test = []
    
    # Load DocVQA data
    docvqa_path = Path('./data/docvqa_training_data.json')
    if docvqa_path.exists():
        print("\n[1/3] Loading DocVQA data...")
        try:
            train, val, test = load_docvqa_data(docvqa_path, max_samples=2000)
            all_train.extend(train)
            all_val.extend(val)
            all_test.extend(test)
            print(f"   Added {len(train)} DocVQA train samples")
            print(f"   Added {len(val)} DocVQA val samples")
            print(f"   Added {len(test)} DocVQA test samples")
        except Exception as e:
            print(f"   Error loading DocVQA: {e}")
    else:
        print("\n[1/3] DocVQA data not found, skipping...")
    
    # Load real InfoVQA data
    infovqa_dir = Path('./data/infovqa')
    print("\n[2/3] Loading Real InfoVQA data...")
    
    if infovqa_dir.exists():
        train, val, test = load_infovqa_data(infovqa_dir, max_samples=3000)
        
        # Add InfoVQA data to combined sets
        all_train.extend(train)
        all_val.extend(val)
        all_test.extend(test)
        
        print(f"\n   Added {len(train)} InfoVQA train samples")
        print(f"   Added {len(val)} InfoVQA val samples")
        print(f"   Added {len(test)} InfoVQA test samples")
    else:
        print(f"   InfoVQA directory not found: {infovqa_dir}")
        print("   Please place InfoVQA annotation files in: data/infovqa/annotations/")
        print("\n   Creating synthetic visual/hybrid data as fallback...")
        
        # Fallback to synthetic data
        from combined_data_loader_fixed import create_synthetic_balanced_data
        synth_train, synth_val, synth_test = create_synthetic_balanced_data(num_samples=3000)
        all_train.extend(synth_train)
        all_val.extend(synth_val)
        all_test.extend(synth_test)
        print(f"   Added {len(synth_train)} synthetic train samples")
        print(f"   Added {len(synth_val)} synthetic val samples")
        print(f"   Added {len(synth_test)} synthetic test samples")
    
    # Shuffle
    random.shuffle(all_train)
    random.shuffle(all_val)
    random.shuffle(all_test)
    
    # Print final distribution
    print("\n[3/3] Final combined distribution:")
    
    for name, data in [("Train", all_train), ("Val", all_val), ("Test", all_test)]:
        counts = defaultdict(int)
        for t, _ in data:
            counts[t] += 1
        print(f"\n   {name}: {len(data)} samples")
        if len(data) > 0:
            print(f"      Text: {counts.get('text', 0)} ({counts.get('text', 0)/len(data)*100:.1f}%)")
            print(f"      Visual: {counts.get('visual', 0)} ({counts.get('visual', 0)/len(data)*100:.1f}%)")
            print(f"      Hybrid: {counts.get('hybrid', 0)} ({counts.get('hybrid', 0)/len(data)*100:.1f}%)")
    
    # Save combined data
    output = {
        'train': all_train,
        'val': all_val,
        'test': all_test,
        'type_mapping': {'text': 0, 'visual': 1, 'hybrid': 2},
        'sources': {
            'docvqa_samples': len([t for t, _ in all_train if t in ['text', 'visual', 'hybrid']]),
            'infovqa_samples': len([t for t, _ in all_train if t in ['text', 'visual', 'hybrid']]) - 2000  # approximate
        }
    }
    
    output_path = Path('./data/combined_training_data.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✅ Combined data saved to: {output_path}")
    print(f"   Total train: {len(all_train)}")
    print(f"   Total val: {len(all_val)}")
    print(f"   Total test: {len(all_test)}")
    
    return output

if __name__ == "__main__":
    prepare_combined_data()