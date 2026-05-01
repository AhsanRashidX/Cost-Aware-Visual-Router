"""
Create best possible balanced dataset using all available visual/hybrid samples
"""

import json
import random
from pathlib import Path
from collections import defaultdict

def classify_question_improved(question: str) -> str:
    """Improved classification for InfoVQA questions"""
    q_lower = question.lower()
    
    # Strong visual indicators
    visual_patterns = [
        'where is', 'located', 'position', 'placement', 'layout',
        'color', 'shape', 'size', 'appearance', 'look like',
        'what does the (diagram|chart|graph|figure|image|infographic) show',
        'what is (shown|depicted|illustrated|displayed)',
        'arrangement of', 'placed', 'situated'
    ]
    
    # Strong hybrid indicators
    hybrid_patterns = [
        'why', 'how (does|do|is|are|can)', 'explain', 'compare', 'contrast',
        'difference between', 'relationship between', 'trend', 'pattern',
        'what can (you|we) (infer|conclude|learn|determine)',
        'what information', 'what insight', 'what conclusion',
        'what does this (mean|indicate|suggest)', 'interpret'
    ]
    
    # Check hybrid first
    for pattern in hybrid_patterns:
        if pattern in q_lower:
            return 'hybrid'
    
    # Check visual
    for pattern in visual_patterns:
        if pattern in q_lower:
            return 'visual'
    
    return 'text'

def load_infovqa_data(data_dir: Path):
    """Load and classify all InfoVQA data"""
    annotations_dir = data_dir / 'annotations'
    
    text_samples = []
    visual_samples = []
    hybrid_samples = []
    
    for split in ['train', 'val', 'test']:
        filepath = annotations_dir / f'{split}.json'
        if not filepath.exists():
            continue
        
        print(f"Loading {split} from: {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        items = data.get('data', [])
        
        for item in items:
            question = item.get('question', '')
            if not question:
                continue
            
            q_type = classify_question_improved(question)
            
            if q_type == 'text':
                text_samples.append((q_type, question))
            elif q_type == 'visual':
                visual_samples.append((q_type, question))
            else:
                hybrid_samples.append((q_type, question))
    
    return text_samples, visual_samples, hybrid_samples

def main():
    print("="*60)
    print("Creating Best Possible Balanced Dataset")
    print("="*60)
    
    # Load InfoVQA data
    print("\n[1/3] Loading InfoVQA data...")
    infovqa_dir = Path('./data/infovqa')
    
    if not infovqa_dir.exists():
        print(f"   ❌ InfoVQA directory not found: {infovqa_dir}")
        return
    
    text_samples, visual_samples, hybrid_samples = load_infovqa_data(infovqa_dir)
    
    print(f"\n   Available InfoVQA samples:")
    print(f"      Text: {len(text_samples)}")
    print(f"      Visual: {len(visual_samples)}")
    print(f"      Hybrid: {len(hybrid_samples)}")
    
    # Use ALL available visual and hybrid samples
    max_visual = len(visual_samples)
    max_hybrid = len(hybrid_samples)
    
    print(f"\n   Using all available visual/hybrid samples:")
    print(f"      Visual: {max_visual}")
    print(f"      Hybrid: {max_hybrid}")
    
    # Sample equal number of text samples (to match visual)
    # But don't exceed available text samples
    target_per_type = min(max_visual, max_hybrid, len(text_samples))
    
    print(f"\n   Target per type: {target_per_type}")
    
    # Sample equally from each type
    sampled_text = random.sample(text_samples, target_per_type)
    sampled_visual = visual_samples  # Use all visual samples
    sampled_hybrid = hybrid_samples  # Use all hybrid samples
    
    # Combine balanced data
    balanced_data = sampled_text + sampled_visual + sampled_hybrid
    random.shuffle(balanced_data)
    
    # Split into train/val/test (70/15/15)
    total = len(balanced_data)
    train_size = int(total * 0.7)
    val_size = int(total * 0.15)
    
    train_data = balanced_data[:train_size]
    val_data = balanced_data[train_size:train_size + val_size]
    test_data = balanced_data[train_size + val_size:]
    
    print(f"\n[2/3] Balanced dataset created:")
    print(f"   Total: {total} samples")
    print(f"   Train: {len(train_data)}")
    print(f"   Val: {len(val_data)}")
    print(f"   Test: {len(test_data)}")
    
    # Print distribution
    print("\n[3/3] Final distribution:")
    
    for name, data in [("Train", train_data), ("Val", val_data), ("Test", test_data)]:
        counts = defaultdict(int)
        for t, _ in data:
            counts[t] += 1
        print(f"\n   {name}: {len(data)} samples")
        print(f"      Text: {counts.get('text', 0)} ({counts.get('text', 0)/len(data)*100:.1f}%)")
        print(f"      Visual: {counts.get('visual', 0)} ({counts.get('visual', 0)/len(data)*100:.1f}%)")
        print(f"      Hybrid: {counts.get('hybrid', 0)} ({counts.get('hybrid', 0)/len(data)*100:.1f}%)")
    
    # Save balanced dataset
    output = {
        'train': train_data,
        'val': val_data,
        'test': test_data,
        'type_mapping': {'text': 0, 'visual': 1, 'hybrid': 2},
        'stats': {
            'total_samples': total,
            'visual_samples_used': len(sampled_visual),
            'hybrid_samples_used': len(sampled_hybrid),
            'text_samples_used': len(sampled_text)
        }
    }
    
    output_path = Path('./data/balanced_training_data_v2.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✅ Balanced dataset saved to: {output_path}")
    print(f"\n📊 Summary: Used all {len(sampled_visual)} visual and {len(sampled_hybrid)} hybrid samples")
    print(f"   Text samples limited to {target_per_type} for balance")

if __name__ == "__main__":
    main()