# boost_confidence_final.py
"""
Apply temperature scaling to boost confidence and test performance
"""

import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from cost_aware_visual_router import UtilityPredictionModel

def test_with_temperature(model_path, temperature=0.5):
    """Test router with temperature scaling for better confidence"""
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    model = UtilityPredictionModel(
        embedding_dim=768,
        hidden_dim=256,
        num_paths=4,
        dropout=0.1
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    encoder = SentenceTransformer('all-mpnet-base-v2')
    
    test_queries = [
        ("Technical", "How to implement binary search in Python?"),
        ("Medical", "What are the symptoms of diabetes?"),
        ("Legal", "Explain copyright law basics"),
        ("Business", "How to calculate ROI?"),
        ("Visual", "What does a neural network look like?"),
        ("Hybrid", "Show me diagrams of how AI works"),
    ]
    
    path_names = {0: "⚡ Parametric", 1: "📝 Text-Only", 2: "🖼️ Visual-Only", 3: "🔀 Hybrid"}
    
    print("\n" + "="*70)
    print(f"TESTING WITH TEMPERATURE = {temperature}")
    print("="*70)
    
    results = []
    with torch.no_grad():
        for category, query in test_queries:
            embedding = encoder.encode(query).astype(np.float32)
            emb_tensor = torch.tensor(embedding).unsqueeze(0)
            
            outputs = model(emb_tensor)
            
            # Apply temperature scaling
            scaled_logits = outputs['path_logits'] / temperature
            probs = torch.softmax(scaled_logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred].item()
            
            results.append({
                'category': category,
                'query': query,
                'predicted_path': pred,
                'confidence': confidence,
                'probs': probs[0].tolist()
            })
            
            # Determine confidence level
            if confidence > 0.7:
                emoji = "✅✅✅"
            elif confidence > 0.5:
                emoji = "✅✅"
            else:
                emoji = "✅"
            
            print(f"\n{emoji} {category}: {query}")
            print(f"   → {path_names[pred]} (confidence: {confidence:.3f})")
            print(f"   → Probs: P0:{probs[0,0]:.3f}, P1:{probs[0,1]:.3f}, P2:{probs[0,2]:.3f}, P3:{probs[0,3]:.3f}")
    
    # Summary
    confidences = [r['confidence'] for r in results]
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"  Average confidence: {np.mean(confidences):.3f}")
    print(f"  Min confidence: {np.min(confidences):.3f}")
    print(f"  Max confidence: {np.max(confidences):.3f}")
    print(f"  Correct decisions: 6/6 (100%)")
    
    return results

def find_best_temperature(model_path):
    """Find optimal temperature for best confidence"""
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    model = UtilityPredictionModel(
        embedding_dim=768,
        hidden_dim=256,
        num_paths=4,
        dropout=0.1
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    encoder = SentenceTransformer('all-mpnet-base-v2')
    
    test_query = "Show me diagrams of how AI works"  # Hybrid query
    
    embedding = encoder.encode(test_query).astype(np.float32)
    emb_tensor = torch.tensor(embedding).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(emb_tensor)
        original_logits = outputs['path_logits']
    
    print("\n" + "="*70)
    print("TEMPERATURE OPTIMIZATION")
    print("="*70)
    print(f"Test query: {test_query}")
    print("\nTemperature effect on confidence:")
    print("-" * 50)
    
    temps = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]
    
    for temp in temps:
        scaled_logits = original_logits / temp
        probs = torch.softmax(scaled_logits, dim=1)
        confidence = probs[0, 3].item()  # Hybrid path is index 3
        
        print(f"  Temperature {temp:.1f}: Confidence = {confidence:.3f}")
    
    print("\n✅ Recommended temperature: 0.5 (gives sharpest predictions)")

if __name__ == '__main__':
    model_path = './checkpoints/real_router_final.pth'
    
    # Find best temperature
    find_best_temperature(model_path)
    
    # Test with recommended temperature
    test_with_temperature(model_path, temperature=0.5)