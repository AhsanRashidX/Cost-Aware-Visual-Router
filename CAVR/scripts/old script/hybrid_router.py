# scripts/hybrid_router.py
"""
Hybrid router: ML model + rule-based corrections
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from cost_aware_visual_router import UtilityPredictionModel


class HybridRouter:
    """
    Hybrid router: ML model with rule-based overrides
    """
    
    def __init__(self, router_model_path='./checkpoints/real_router_final.pth'):
        print("Loading Hybrid Router...")
        
        # Load embedding model
        self.tokenizer = AutoTokenizer.from_pretrained("./models/all-mpnet-base-v2")
        self.embedding_model = AutoModel.from_pretrained("./models/all-mpnet-base-v2")
        self.embedding_model.eval()
        
        # Load routing model
        checkpoint = torch.load(router_model_path, map_location='cpu', weights_only=False)
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
    
    def encode_query(self, query: str) -> np.ndarray:
        inputs = self.tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.squeeze().numpy().astype(np.float32)
    
    def route_and_retrieve(self, query: str):
        # Rule-based overrides (fix the visual/hybrid detection)
        q_lower = query.lower()
        
        # Override: Visual questions (where/located)
        if 'where' in q_lower or 'located' in q_lower:
            decision = 2  # visual
            confidence = 0.85
            print(f"   [RULE OVERRIDE: visual]")
        # Override: Hybrid questions (why/how)
        elif q_lower.startswith(('why', 'how')):
            decision = 3  # hybrid
            confidence = 0.80
            print(f"   [RULE OVERRIDE: hybrid]")
        else:
            # Use ML model for text
            embedding = self.encode_query(query)
            emb_tensor = torch.tensor(embedding).unsqueeze(0)
            
            with torch.no_grad():
                outputs = self.router(emb_tensor)
                scaled_logits = outputs['path_logits'] / self.temperature
                probs = torch.softmax(scaled_logits, dim=1)
                decision = torch.argmax(probs, dim=1).item()
                confidence = probs[0, decision].item()
        
        path_name = self.path_names[decision]
        cost_map = {0: 0.0001, 1: 0.0005, 2: 0.01, 3: 0.0105}
        
        return {
            'path_name': path_name,
            'decision': decision,
            'confidence': confidence,
            'retrieval_result': {'cost': cost_map[decision], 'latency_ms': 50}
        }