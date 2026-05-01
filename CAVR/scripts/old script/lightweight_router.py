"""
Lightweight router wrapper for evaluation (avoids loading heavy ColPali model)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from sentence_transformers import SentenceTransformer


class LightweightRouter:
    """
    Lightweight version of router for evaluation
    Loads only the routing model, not ColPali
    """
    
    def __init__(self, router_model_path='./checkpoints/real_router_final.pth'):
        print("   Loading routing model...")
        
        # Load encoder
        self.encoder = SentenceTransformer('all-mpnet-base-v2')
        
        # Load router model
        from cost_aware_visual_router import UtilityPredictionModel
        
        checkpoint = torch.load(router_model_path, weights_only=False)
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
        print("   ✅ Router loaded successfully")
    
    def route_and_retrieve(self, query: str):
        """Route query without actual retrieval (for evaluation)"""
        
        # Get embedding
        embedding = self.encoder.encode(query).astype(np.float32)
        emb_tensor = torch.tensor(embedding).unsqueeze(0)
        
        with torch.no_grad():
            outputs = self.router(emb_tensor)
            scaled_logits = outputs['path_logits'] / self.temperature
            probs = torch.softmax(scaled_logits, dim=1)
            decision = torch.argmax(probs, dim=1).item()
            confidence = probs[0, decision].item()
        
        path_name = self.path_names[decision]
        
        # Estimate cost based on path
        if decision == 0:
            cost = 0.0001
        elif decision == 1:
            cost = 0.0005
        elif decision == 2:
            cost = 0.01
        else:
            cost = 0.0105
        
        return {
            'path_name': path_name,
            'confidence': confidence,
            'retrieval_result': {'cost': cost}
        }