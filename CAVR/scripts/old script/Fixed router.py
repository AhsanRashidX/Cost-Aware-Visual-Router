"""
Fixed router with local sentence transformer model
No hanging - uses pre-downloaded model
"""

import sys
import os
from pathlib import Path

# Set environment variables BEFORE importing sentence_transformers
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

# Set cache directory
cache_dir = Path('./cache/sentence_transformers')
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ['SENTENCE_TRANSFORMERS_HOME'] = str(cache_dir)

print("="*60)
print("Loading Fixed Router")
print("="*60)

# Import modules
print("\n[1/4] Importing torch...")
import torch
print(f"   ✓ torch {torch.__version__}")

print("\n[2/4] Importing numpy...")
import numpy as np
print(f"   ✓ numpy {np.__version__}")

print("\n[3/4] Importing sentence_transformers...")
try:
    from sentence_transformers import SentenceTransformer
    print(f"   ✓ sentence_transformers imported")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

print("\n[4/4] Loading models...")

class FixedRouter:
    """
    Router that uses local pre-downloaded sentence transformer model
    """
    
    def __init__(self, router_model_path='./checkpoints/real_router_final.pth'):
        # Load sentence encoder
        print("   Loading sentence encoder...")
        local_model_path = Path('./models/all-mpnet-base-v2')
        
        if local_model_path.exists():
            print(f"   Using local model from: {local_model_path}")
            self.encoder = SentenceTransformer(str(local_model_path))
        else:
            print("   Using cached model...")
            self.encoder = SentenceTransformer('all-mpnet-base-v2')
        
        # Load routing model
        print("   Loading routing model...")
        from cost_aware_visual_router import UtilityPredictionModel
        
        checkpoint_path = Path(router_model_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        self.router = UtilityPredictionModel(768, 256, 4, 0.1)
        self.router.load_state_dict(checkpoint['model_state_dict'])
        self.router.eval()
        
        self.path_names = {
            0: "⚡ Parametric (LLM only)",
            1: "📝 Text-Only Retrieval",
            2: "🖼️ Visual-Only (ColPali)",
            3: "🔀 Hybrid (Text + ColPali)"
        }
        
        self.temperature = 0.7
        print("   ✅ Router loaded successfully!")
    
    def encode_query(self, query: str) -> np.ndarray:
        """Encode query using sentence transformer"""
        return self.encoder.encode(query).astype(np.float32)
    
    def route_and_retrieve(self, query: str):
        """Route query to appropriate path"""
        
        # Get embedding
        embedding = self.encode_query(query)
        emb_tensor = torch.tensor(embedding).unsqueeze(0)
        
        with torch.no_grad():
            outputs = self.router(emb_tensor)
            scaled_logits = outputs['path_logits'] / self.temperature
            probs = torch.softmax(scaled_logits, dim=1)
            decision = torch.argmax(probs, dim=1).item()
            confidence = probs[0, decision].item()
        
        path_name = self.path_names[decision]
        
        # Estimate cost
        cost_map = {0: 0.0001, 1: 0.0005, 2: 0.01, 3: 0.0105}
        
        return {
            'path_name': path_name,
            'decision': decision,
            'confidence': confidence,
            'retrieval_result': {
                'cost': cost_map[decision],
                'latency_ms': 10
            }
        }


# Test the router
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Testing Fixed Router")
    print("="*60)
    
    try:
        router = FixedRouter()
        
        test_queries = [
            "What is Python programming?",
            "What does a neural network diagram look like?",
            "Explain how a car engine works with diagram"
        ]
        
        print("\nTesting queries:")
        for q in test_queries:
            result = router.route_and_retrieve(q)
            print(f"\n   Query: {q}")
            print(f"   Decision: {result['path_name']}")
            print(f"   Confidence: {result['confidence']:.3f}")
        
        print("\n✅ Router test successful!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()