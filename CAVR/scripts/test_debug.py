import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

print(f"Project root: {Path(__file__).parent}")

try:
    from utils.cost_tracker import RealCostTracker
    print("✅ Import successful!")
    tracker = RealCostTracker()
    print("✅ Cost tracker created successfully!")
except Exception as e:
    print(f"❌ Import failed: {e}")