"""
Inspect checkpoint to understand model architecture
"""

import torch
from pathlib import Path

print("="*60)
print("Inspecting Checkpoint")
print("="*60)

checkpoint_path = Path('./checkpoints/docvqa_router_fixed.pth')
if not checkpoint_path.exists():
    checkpoint_path = Path('./checkpoints/docvqa_router_final.pth')
if not checkpoint_path.exists():
    print("No checkpoint found!")
    exit()

print(f"\nLoading checkpoint: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

print(f"\nCheckpoint keys: {list(checkpoint.keys())}")

if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
    print(f"\nModel state dict keys (first 20):")
    for i, key in enumerate(list(state_dict.keys())[:20]):
        print(f"   {key}: {state_dict[key].shape}")
    
    print(f"\nTotal layers: {len(state_dict.keys())}")
    
    # Check for architecture clues
    if 'shared.0.weight' in state_dict:
        print("\n✅ Model appears to be a SimpleClassifier (shared + classifier)")
        print("   Architecture: shared layers -> classifier")
    elif 'feature_extractor.0.weight' in state_dict:
        print("\n✅ Model appears to be UtilityPredictionModel")
        print("   Architecture: feature_extractor -> path_classifier + utility_heads + confidence_head")

if 'config' in checkpoint:
    print(f"\nConfig: {checkpoint['config']}")

print("\n" + "="*60)