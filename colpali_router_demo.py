"""
Complete ColPali-Enabled Router Demo
Shows routing + actual retrieval with ColPali
"""

import torch
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from cost_aware_visual_router import UtilityPredictionModel
from byaldi import RAGMultiModalModel
import time
from PIL import Image

# ============================================
# COL PALI RETRIEVER CLASS
# ============================================

class ColPaliRetriever:
    """Document retriever using ColPali for visual understanding"""
    
    def __init__(self, model_name: str = "vidore/colpali-v1.3-hf"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🚀 Initializing ColPali Retriever on {self.device}...")
        
        # Initialize the ColPali model
        self.model = RAGMultiModalModel.from_pretrained(model_name)
        self.current_index_name = None
        
        # Cost structure (matches your visual-only path)
        self.cost_per_query = 0.01  # $0.01 per visual query
        
    def index_documents(self, documents_path: str, index_name: str = "my_docs"):
        """
        Index your document collection
        documents_path: Path to folder containing PDFs or images
        """
        print(f"📑 Indexing documents from {documents_path}...")
        self.model.index(
            input_path=documents_path,
            index_name=index_name,
            overwrite=True
        )
        self.current_index_name = index_name
        print(f"✅ Indexed {index_name} successfully")
        
    def retrieve(self, query: str, k: int = 5) -> dict:
        """Retrieve relevant document pages for a query"""
        start_time = time.time()
        
        if self.current_index_name is None:
            raise ValueError("No index loaded. Run index_documents() first.")
        
        # Search with ColPali
        results = self.model.search(query, k=k)
        
        latency = (time.time() - start_time) * 1000
        
        # Format results
        retrieved_docs = []
        similarities = []
        for result in results:
            retrieved_docs.append({
                'doc_id': result.doc_id,
                'page_num': result.page_num,
                'score': result.score,
                'content': result.text if hasattr(result, 'text') else None
            })
            similarities.append(result.score)
        
        return {
            'answer': f"Found {len(results)} relevant pages. Top score: {results[0].score:.3f}" if results else "No results",
            'latency_ms': latency,
            'cost': self.cost_per_query,
            'retrieved_docs': retrieved_docs,
            'similarities': similarities
        }


# ============================================
# COMPLETE ROUTER WITH COL PALI
# ============================================

class CompleteVisualRouter:
    """Full router with ColPali for visual retrieval"""
    
    def __init__(self, router_model_path='./checkpoints/real_router_final.pth'):
        # Load your trained router
        print("Loading trained router...")
        checkpoint = torch.load(router_model_path, weights_only=False)
        self.router = UtilityPredictionModel(768, 256, 4, 0.1)
        self.router.load_state_dict(checkpoint['model_state_dict'])
        self.router.eval()
        
        # Load encoder for routing
        self.encoder = SentenceTransformer('all-mpnet-base-v2')
        
        # Initialize ColPali for visual retrieval
        self.colpali = ColPaliRetriever()
        
        # Path mapping
        self.path_names = {
            0: "⚡ Parametric (LLM only)",
            1: "📝 Text-Only Retrieval",
            2: "🖼️ Visual-Only (ColPali)",
            3: "🔀 Hybrid (Text + ColPali)"
        }
        
        self.temperature = 0.5
        
    def route_and_retrieve(self, query: str):
        """
        Step 1: Router decides which path to use
        Step 2: Execute actual retrieval based on decision
        """
        
        # STEP 1: Get router decision
        embedding = self.encoder.encode(query).astype(np.float32)
        emb_tensor = torch.tensor(embedding).unsqueeze(0)
        
        with torch.no_grad():
            outputs = self.router(emb_tensor)
            scaled_logits = outputs['path_logits'] / self.temperature
            probs = torch.softmax(scaled_logits, dim=1)
            decision = torch.argmax(probs, dim=1).item()
            confidence = probs[0, decision].item()
        
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"Router Decision: {self.path_names[decision]} (confidence: {confidence:.3f})")
        print(f"{'='*60}")
        
        # STEP 2: Execute actual retrieval based on decision
        if decision == 0:  # Parametric
            result = self._parametric_retrieval(query)
        elif decision == 1:  # Text-Only
            result = self._text_retrieval(query)
        elif decision == 2:  # Visual-Only (ColPali)
            result = self.colpali.retrieve(query)
        else:  # Hybrid
            text_result = self._text_retrieval(query)
            visual_result = self.colpali.retrieve(query)
            result = self._merge_results(text_result, visual_result)
        
        return {
            'query': query,
            'decision': decision,
            'path_name': self.path_names[decision],
            'confidence': confidence,
            'retrieval_result': result
        }
    
    def _parametric_retrieval(self, query):
        """Simulate LLM parametric knowledge"""
        return {
            'answer': f"[Parametric] Answering from LLM knowledge: {query}",
            'latency_ms': 50,
            'cost': 0.0001,
            'retrieved_docs': []
        }
    
    def _text_retrieval(self, query):
        """Simulate text-only retrieval"""
        return {
            'answer': f"[Text Retrieval] Found documents about: {query}",
            'latency_ms': 100,
            'cost': 0.0005,
            'retrieved_docs': [{'id': 1, 'text': 'Sample document'}],
            'similarities': [0.85]
        }
    
    def _merge_results(self, text_result, visual_result):
        """Merge text and visual results"""
        return {
            'answer': f"[Hybrid] {text_result['answer']} + {visual_result['answer']}",
            'latency_ms': text_result['latency_ms'] + visual_result['latency_ms'],
            'cost': text_result['cost'] + visual_result['cost'],
            'retrieved_docs': text_result.get('retrieved_docs', []) + visual_result.get('retrieved_docs', []),
            'similarities': text_result.get('similarities', []) + visual_result.get('similarities', [])
        }


# ============================================
# DEMO FUNCTION
# ============================================

def demo_colpali_router():
    """Demonstrate the complete ColPali-enabled router"""
    
    print("="*70)
    print("COL PALI-ENABLED VISUAL ROUTER DEMO")
    print("="*70)
    
    # Initialize the complete router
    router = CompleteVisualRouter()
    
    # Example: Index your documents first (do this once)
    # router.colpali.index_documents("./my_documents_folder", "my_docs")
    
    # Test queries that should trigger visual retrieval
    test_queries = [
        "What does a neural network architecture diagram look like?",
        "Show me examples of Art Deco architecture",
        "Explain how a car engine works with diagrams",
        "What does the Eiffel Tower look like?",
        "How to tie a tie with pictures",
    ]
    
    results = []
    for query in test_queries:
        result = router.route_and_retrieve(query)
        results.append(result)
        
        # Display retrieval result
        retrieval = result['retrieval_result']
        print(f"\n📄 Retrieval Result:")
        print(f"   Answer: {retrieval['answer'][:150]}...")
        print(f"   Latency: {retrieval['latency_ms']:.1f}ms")
        print(f"   Cost: ${retrieval['cost']:.5f}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for r in results:
        print(f"\n{r['query'][:50]}...")
        print(f"  → {r['path_name']} (conf: {r['confidence']:.3f})")
    
    return results


# ============================================
# INTEGRATE WITH YOUR EXISTING TEST SCRIPT
# ============================================

def test_with_colpali_integration(queries_file="test_queries.txt"):
    """
    Enhanced version of your test_router that actually executes
    the retrieval based on router decisions
    """
    
    # Load your existing test queries
    from test_real_router import load_queries_from_file
    
    test_queries = load_queries_from_file(queries_file)
    
    if not test_queries:
        print("No test queries found!")
        return
    
    # Initialize the complete router
    router = CompleteVisualRouter()
    
    print("\n" + "="*70)
    print("TESTING ROUTER WITH ACTUAL RETRIEVAL")
    print("="*70)
    
    results = []
    for category, query in test_queries[:10]:  # Test first 10
        result = router.route_and_retrieve(query)
        results.append(result)
        
        print(f"\n[{category}] {query[:60]}...")
        print(f"  → Decision: {result['path_name']}")
        print(f"  → Confidence: {result['confidence']:.3f}")
    
    return results


if __name__ == '__main__':
    # Run the demo
    demo_colpali_router()
    
    # Or run with your test queries
    # test_with_colpali_integration()