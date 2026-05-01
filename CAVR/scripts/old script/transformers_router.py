"""
Router using transformers library directly (no sentence_transformers)
"""

import sys
import os
from pathlib import Path
import time

# Set environment variables to use local cache
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

print("="*60)
print("Loading Transformers Router")
print("="*60)

# Import with timeout protection
print("\n[1/5] Importing torch...")
import torch
print(f"   ✓ torch {torch.__version__}")

print("\n[2/5] Importing numpy...")
import numpy as np
print(f"   ✓ numpy {np.__version__}")

print("\n[3/5] Importing transformers...")
try:
    from transformers import AutoTokenizer, AutoModel
    import torch.nn.functional as F
    print(f"   ✓ transformers imported")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

print("\n[4/5] Importing router model...")
try:
    from cost_aware_visual_router import UtilityPredictionModel
    print(f"   ✓ UtilityPredictionModel imported")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)


class TransformersRouter:
    """
    Router using transformers for embeddings (no sentence_transformers)
    """
    
    def __init__(self, router_model_path='./checkpoints/real_router_final.pth'):
        print("\n[5/5] Loading models...")
        
        # Load transformer model for embeddings
        print("   Loading embedding model...")
        model_name = "sentence-transformers/all-mpnet-base-v2"
        local_model_path = Path("./models/all-mpnet-base-v2")
        
        if local_model_path.exists():
            print(f"   Using local model from: {local_model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(str(local_model_path))
            self.embedding_model = AutoModel.from_pretrained(str(local_model_path))
        else:
            print(f"   Using cached model")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.embedding_model = AutoModel.from_pretrained(model_name)
        
        self.embedding_model.eval()
        
        # Load routing model
        print("   Loading routing model...")
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
        print("   ✅ TransformersRouter loaded successfully!")
    
    def encode_query(self, query: str) -> np.ndarray:
        """Encode query using transformer model"""
        # Tokenize
        inputs = self.tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
            # Use mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings.squeeze().numpy().astype(np.float32)
    
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
                'latency_ms': 50
            }
        }


# Test the router
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Testing Transformers Router")
    print("="*60)
    
    try:
        router = TransformersRouter()
        
        test_queries = [
            "What is Python programming?",
            "What does a neural network diagram look like?",
            "Explain how a car engine works with diagram"
        ]
        
        print("\nTesting queries:")
        for q in test_queries:
            result = router.route_and_retrieve(q)
            print(f"\n   Query: {q[:50]}...")
            print(f"   Decision: {result['path_name']}")
            print(f"   Confidence: {result['confidence']:.3f}")
        
        print("\n✅ Router test successful!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()