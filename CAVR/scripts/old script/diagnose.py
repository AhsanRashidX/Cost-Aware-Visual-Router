# save as scripts/diagnose.py
import os
import sys
from pathlib import Path

print("="*60)
print("DIAGNOSTIC - Checking Environment")
print("="*60)

# Check current directory
print(f"\n1. Current directory: {os.getcwd()}")

# Check if files exist
paths_to_check = [
    "./data/docvqa/annotations/train.json",
    "./data/docvqa/annotations/val.json", 
    "./data/docvqa/annotations/test.json",
    "./data/docvqa/images",
    "./data/infovqa/annotations/train.json",
    "./data/infovqa/annotations/val.json",
    "./colpali_router_demo.py",
]

print("\n2. Checking file paths:")
for path in paths_to_check:
    exists = Path(path).exists()
    status = "✅" if exists else "❌"
    print(f"   {status} {path}")

# Check if we can import router
print("\n3. Testing router import:")
try:
    from colpali_router_demo import CompleteVisualRouter
    print("   ✅ CompleteVisualRouter imported successfully")
except Exception as e:
    print(f"   ❌ Import failed: {e}")

print("\n" + "="*60)