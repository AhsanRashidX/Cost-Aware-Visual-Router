# # test_real_router.py
# """
# Test your trained router on real multimodal queries
# """

# import torch
# import numpy as np
# from PIL import Image
# from sentence_transformers import SentenceTransformer
# from cost_aware_visual_router import UtilityPredictionModel

# def test_router():
#     # Load trained model
#     checkpoint = torch.load('./checkpoints/real_router_final.pth', weights_only=False)
    
#     model = UtilityPredictionModel(
#         embedding_dim=768,
#         hidden_dim=256,
#         num_paths=4,
#         dropout=0.1
#     )
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model.eval()
    
#     # Load encoder
#     encoder = SentenceTransformer('all-mpnet-base-v2')
    
#     # Test queries from different domains
#     test_queries = test_queries = [
#         # 🟢 TECHNICAL/PROGRAMMING (Should prefer Text-Only/Parametric)
#         ("Technical", "How do I fix 'undefined is not a function' in JavaScript?"),
#         ("Technical", "What's the difference between let and const in JavaScript?"),
#         ("Technical", "How to connect to PostgreSQL database in Python?"),
#         ("Technical", "Why is my React component not re-rendering?"),
#         ("Technical", "How to handle errors in async/await?"),
#         ("Technical", "What is dependency injection and why use it?"),
#         ("Technical", "How to optimize SQL query performance?"),
#         ("Technical", "Explain the difference between TCP and UDP"),
        
#         # 🟡 MEDICAL/HEALTHCARE (Should prefer Visual/Text combination)
#         ("Medical", "What are the early warning signs of a heart attack?"),
#         ("Medical", "How to check blood pressure at home?"),
#         ("Medical", "What does a melanoma skin cancer look like?"),
#         ("Medical", "How to perform CPR correctly?"),
#         ("Medical", "What are the side effects of COVID-19 vaccine?"),
#         ("Medical", "How to interpret an ECG reading?"),
#         ("Medical", "What does a healthy diet look like for diabetics?"),
        
#         # 🔵 LEGAL (Should prefer Text-Only/Legal documents)
#         ("Legal", "What are my rights if I'm pulled over by police?"),
#         ("Legal", "How to write a legally binding contract?"),
#         ("Legal", "What is the difference between copyright and trademark?"),
#         ("Legal", "How to file for divorce without a lawyer?"),
#         ("Legal", "What are tenant rights for security deposits?"),
#         ("Legal", "How to create a will that is legally valid?"),
#         ("Legal", "What is employment at will and what does it mean?"),
        
#         # 🟠 BUSINESS/MARKETING (Should prefer Text-Only/Business docs)
#         ("Business", "How to calculate break-even point for a startup?"),
#         ("Business", "What's the best way to find product-market fit?"),
#         ("Business", "How to create a marketing budget for small business?"),
#         ("Business", "What metrics should I track for SaaS business?"),
#         ("Business", "How to price a product for maximum profit?"),
#         ("Business", "How to write a business plan that investors like?"),
#         ("Business", "What is customer churn and how to reduce it?"),
        
#         # 🟣 VISUAL/IMAGE-RELATED (Should prefer Visual-Only)
#         ("Visual", "Show me what a cat scan image looks like"),
#         ("Visual", "What does a wiring diagram for a 3-way switch look like?"),
#         ("Visual", "Show me examples of Art Deco architecture in NYC"),
#         ("Visual", "What does a positive pregnancy test look like?"),
#         ("Visual", "Show me different types of clouds with pictures"),
#         ("Visual", "What does a skin rash from poison ivy look like?"),
#         ("Visual", "Show me how to tie a tie step by step with images"),
        
#         # 🔄 HYBRID (Needs both text and visuals)
#         ("Hybrid", "Explain how a car engine works with diagrams"),
#         ("Hybrid", "How to bake sourdough bread with step-by-step photos"),
#         ("Hybrid", "What does the human digestive system look like with labels?"),
#         ("Hybrid", "Show me how to do yoga poses with instructions and images"),
#         ("Hybrid", "Explain the water cycle with visual examples"),
#         ("Hybrid", "How to change a tire with pictures and instructions"),
#         ("Hybrid", "What does a neural network architecture diagram look like?"),
        
#         # 📚 EDUCATIONAL (Mixed)
#         ("Educational", "What does the periodic table look like?"),
#         ("Educational", "How to solve quadratic equations step by step?"),
#         ("Educational", "Show me the structure of a plant cell with diagram"),
#         ("Educational", "Explain photosynthesis with visual guide"),
#         ("Educational", "What are the parts of a microscope with labels?"),
        
#         # 💼 REAL-WORLD SCENARIOS
#         ("RealWorld", "I need to write a resignation letter, what should it say?"),
#         ("RealWorld", "How to fix a leaking faucet with pictures?"),
#         ("RealWorld", "What does poison ivy look like?"),
#         ("RealWorld", "How to calculate my monthly mortgage payment?"),
#         ("RealWorld", "What are the symptoms of food poisoning?"),
        
#         # ⚠️ EDGE CASES (Short queries, typos, slang)
#         ("Short", "Python async"),
#         ("Short", "chest pain causes"),
#         ("Short", "copyright fair use"),
#         ("Short", "ROI formula"),
#         ("Typo", "how to impliment binary search in pyhton"),
#         ("Typo", "what are symtoms of diabites"),
#         ("Slang", "best way to get ROI in real estate"),
#         ("Slang", "how to not get sued for copyright"),
#     ]
    
#     print("\n" + "="*70)
#     print("TESTING TRAINED ROUTER ON REAL DATA")
#     print("="*70)
    
#     path_names = {0: "Parametric", 1: "Text-Only", 2: "Visual-Only", 3: "Hybrid"}
    
#     with torch.no_grad():
#         for domain, query in test_queries:
#             embedding = encoder.encode(query).astype(np.float32)
#             emb_tensor = torch.tensor(embedding).unsqueeze(0)
#             temperature = 0.5
#             outputs = model(emb_tensor)
#             scaled_logits = outputs['path_logits'] / temperature
#             probs = torch.softmax(scaled_logits, dim=1)
#             # probs = torch.softmax(outputs['path_logits'], dim=1)
#             pred = torch.argmax(probs, dim=1).item()
#             confidence = probs[0, pred].item()
            
#             print(f"\n{domain}: {query}")
#             print(f"  → {path_names[pred]} (confidence: {confidence:.3f})")
#             print(f"  → All probs: P0:{probs[0,0]:.3f}, P1:{probs[0,1]:.3f}, P2:{probs[0,2]:.3f}, P3:{probs[0,3]:.3f}")

# if __name__ == '__main__':
#     test_router()

"""
Test your trained router on real multimodal queries
Reads test queries from external text file
"""
from byaldi import RAGMultiModalModel
import time
import torch
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from cost_aware_visual_router import UtilityPredictionModel


# Add this class before your existing functions
class ColPaliRetriever:
    """Minimal ColPali integration"""
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Loading ColPali on {self.device}...")
        self.model = RAGMultiModalModel.from_pretrained("vidore/colpali-v1.3-hf")
    
    def retrieve(self, query, k=5):
        start = time.time()
        # You would need to index documents first
        # results = self.model.search(query, k=k)
        latency = (time.time() - start) * 1000
        return {
            'answer': f"ColPali retrieval for: {query}",
            'latency_ms': latency,
            'cost': 0.01,
            'similarities': [0.9]
        }

def load_queries_from_file(filepath="test_queries.txt"):
    """
    Load test queries from text file.
    Format: Category|Query
    Lines starting with # are ignored
    """
    queries = []
    file_path = Path(filepath)
    
    if not file_path.exists():
        print(f"❌ File not found: {filepath}")
        print(f"   Creating sample file: {filepath}")
        create_sample_queries_file(filepath)
        return []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Parse category and query
            if '|' in line:
                parts = line.split('|', 1)
                category = parts[0].strip()
                query = parts[1].strip()
                queries.append((category, query))
            else:
                print(f"⚠️ Warning: Line {line_num} has invalid format (missing '|'): {line[:50]}")
    
    print(f"✅ Loaded {len(queries)} test queries from {filepath}")
    return queries

def create_sample_queries_file(filepath="test_queries.txt"):
    """Create a sample queries file if none exists"""
    sample_content = """# Sample test queries file
# Format: Category|Query
# Lines starting with # are ignored

# Technical queries
Technical|How to implement binary search in Python?
Technical|What is the difference between SQL and NoSQL?

# Medical queries
Medical|What are the symptoms of diabetes?
Medical|How to perform CPR correctly?

# Legal queries
Legal|Explain copyright law basics
Legal|What are my rights if pulled over?

# Business queries
Business|How to calculate ROI?
Business|What is customer lifetime value?

# Visual queries
Visual|What does a neural network look like?
Visual|Show me examples of Art Deco architecture

# Hybrid queries
Hybrid|Explain how a car engine works with diagrams
Hybrid|How to change a tire with pictures

# Edge cases
Short|Python async
Typo|how to impliment binary search
"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(sample_content)
    print(f"✅ Created sample file: {filepath}")
    print("   Edit this file to add your own test queries")

def test_router(queries_file="test_queries.txt", temperature=0.5, save_results=True,execute_retrieval=False):
    """
    Test trained router on queries from file
    
    Args:
        queries_file: Path to text file containing test queries
        temperature: Temperature scaling factor (lower = sharper predictions)
        save_results: Whether to save results to CSV file
    """
    # Load trained model
    checkpoint_path = './checkpoints/real_router_final.pth'
    if not Path(checkpoint_path).exists():
        print(f"❌ Model not found: {checkpoint_path}")
        print("   Please train the model first using train_real_router.py")
        return
    
    print("Loading trained model...")
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    
    model = UtilityPredictionModel(
        embedding_dim=768,
        hidden_dim=256,
        num_paths=4,
        dropout=0.1
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load encoder
    print("Loading text encoder...")
    encoder = SentenceTransformer('all-mpnet-base-v2')
    
    # Load test queries from file
    print(f"\nLoading test queries from {queries_file}...")
    test_queries = load_queries_from_file(queries_file)
    
    if not test_queries:
        print("❌ No test queries loaded. Exiting.")
        return
    
    print("\n" + "="*70)
    print("TESTING TRAINED ROUTER ON REAL DATA")
    print("="*70)
    
    path_names = {0: "⚡ Parametric", 1: "📝 Text-Only", 2: "🖼️ Visual-Only", 3: "🔀 Hybrid"}
    results = []
    
    # Track statistics by category
    stats = {}
    
    with torch.no_grad():
        for domain, query in test_queries:
            # Get embedding
            embedding = encoder.encode(query).astype(np.float32)
            emb_tensor = torch.tensor(embedding).unsqueeze(0)
            
            # Get prediction with temperature scaling
            outputs = model(emb_tensor)
            scaled_logits = outputs['path_logits'] / temperature
            probs = torch.softmax(scaled_logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred].item()
            
            # Store result
            result = {
                'category': domain,
                'query': query,
                'predicted_path': pred,
                'path_name': path_names[pred],
                'confidence': confidence,
                'p0': probs[0,0].item(),
                'p1': probs[0,1].item(),
                'p2': probs[0,2].item(),
                'p3': probs[0,3].item()
            }
            results.append(result)
            
            # Update statistics
            if domain not in stats:
                stats[domain] = {'count': 0, 'total_confidence': 0, 'paths': {}}
            stats[domain]['count'] += 1
            stats[domain]['total_confidence'] += confidence
            stats[domain]['paths'][pred] = stats[domain]['paths'].get(pred, 0) + 1
            
            # Print result
            if confidence > 0.7:
                emoji = "✅✅✅"
            elif confidence > 0.5:
                emoji = "✅✅"
            else:
                emoji = "✅"
            
            print(f"\n{emoji} [{domain:12}] {query[:70]}...")
            print(f"     → {path_names[pred]} (confidence: {confidence:.3f})")
            print(f"     → Probs: P0:{probs[0,0]:.3f} P1:{probs[0,1]:.3f} P2:{probs[0,2]:.3f} P3:{probs[0,3]:.3f}")
    
    # Print summary statistics
    print("\n" + "="*70)
    print("📊 SUMMARY STATISTICS")
    print("="*70)
    
    for category, data in stats.items():
        avg_conf = data['total_confidence'] / data['count']
        print(f"\n{category}:")
        print(f"  Total queries: {data['count']}")
        print(f"  Avg confidence: {avg_conf:.3f}")
        print(f"  Path distribution: {data['paths']}")
    
    # Save results to CSV if requested
    if save_results:
        import csv
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"router_test_results_{timestamp}.csv"
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['category', 'query', 'predicted_path', 'path_name', 
                         'confidence', 'p0', 'p1', 'p2', 'p3']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\n💾 Results saved to: {output_file}")
    # Initialize ColPali if executing retrieval
    if execute_retrieval:
        colpali = ColPaliRetriever()
        print("✅ ColPali ready for visual retrieval")
    
    # Then in your prediction loop, after getting decision:
    if execute_retrieval and pred == 2:  # Visual path
        visual_result = colpali.retrieve(query)
        print(f"     🔍 Visual result: {visual_result['answer'][:100]}...")
    return results

def test_single_query(query, temperature=0.5):
    """Test a single query (useful for quick testing)"""
    checkpoint = torch.load('./checkpoints/real_router_final.pth', weights_only=False)
    
    model = UtilityPredictionModel(768, 256, 4, 0.1)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    encoder = SentenceTransformer('all-mpnet-base-v2')
    path_names = {0: "Parametric", 1: "Text-Only", 2: "Visual-Only", 3: "Hybrid"}
    
    embedding = encoder.encode(query).astype(np.float32)
    emb_tensor = torch.tensor(embedding).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(emb_tensor)
        scaled_logits = outputs['path_logits'] / temperature
        probs = torch.softmax(scaled_logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred].item()
    
    print(f"\nQuery: {query}")
    print(f"  → {path_names[pred]} (confidence: {confidence:.3f})")
    print(f"  → Probs: P0:{probs[0,0]:.3f}, P1:{probs[0,1]:.3f}, P2:{probs[0,2]:.3f}, P3:{probs[0,3]:.3f}")
    
    return pred, confidence

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test the trained router')
    parser.add_argument('--file', type=str, default='test_queries.txt',
                       help='Path to test queries file (default: test_queries.txt)')
    parser.add_argument('--query', type=str, default=None,
                       help='Test a single query instead of reading from file')
    parser.add_argument('--temp', type=float, default=0.5,
                       help='Temperature for scaling (default: 0.5)')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save results to CSV')
    
    args = parser.parse_args()
    
    if args.query:
        # Test single query
        test_single_query(args.query, temperature=args.temp)
    else:
        # Test from file
        test_router(
            queries_file=args.file,
            temperature=args.temp,
            save_results=not args.no_save
        )