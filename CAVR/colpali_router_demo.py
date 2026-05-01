"""
Complete ColPali-Enabled Router Demo with Sample Documents
"""

import torch
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from cost_aware_visual_router import UtilityPredictionModel
from byaldi import RAGMultiModalModel
import time
from PIL import Image
import requests
from io import BytesIO

# ============================================
# COL PALI RETRIEVER CLASS (FIXED)
# ============================================

class ColPaliRetriever:
    """Document retriever using ColPali for visual understanding"""
    
    def __init__(self, model_name: str = "vidore/colpali-v1.3-hf"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🚀 Initializing ColPali Retriever on {self.device}...")
        
        try:
            # Initialize the ColPali model
            self.model = RAGMultiModalModel.from_pretrained(model_name, verbose=0)
            self.current_index_name = None
            print("✅ ColPali model loaded successfully")
        except Exception as e:
            print(f"⚠️ Error loading ColPali: {e}")
            print("Using fallback mode (simulated retrieval)")
            self.model = None
        
        # Cost structure (matches your visual-only path)
        self.cost_per_query = 0.01  # $0.01 per visual query
        
    def create_sample_index(self, index_name: str = "sample_docs"):
        """Create a sample index with demo documents"""
        print(f"\n📑 Creating sample index '{index_name}'...")
        
        # Create a temporary directory for sample documents
        sample_dir = Path("./sample_documents")
        sample_dir.mkdir(exist_ok=True)
        
        # Create sample text documents
        sample_docs = {
            "neural_networks.txt": """
            Neural Network Architecture
            A neural network consists of layers of interconnected nodes.
            - Input layer: receives data
            - Hidden layers: process information
            - Output layer: produces results
            Common architectures: CNNs for images, RNNs for sequences, Transformers for text.
            """,
            
            "eiffel_tower.txt": """
            Eiffel Tower
            Located in Paris, France. Built in 1889.
            Height: 330 meters. Made of wrought iron.
            One of the most famous landmarks in the world.
            """,
            
            "car_engine.txt": """
            How a Car Engine Works
            Internal combustion engine converts fuel to motion.
            Key parts: pistons, cylinders, crankshaft, spark plugs.
            Four strokes: intake, compression, power, exhaust.
            """,
            
            "art_deco.txt": """
            Art Deco Architecture
            Style from 1920s-1930s. Features geometric shapes, rich colors.
            Examples: Chrysler Building (NYC), Empire State Building.
            """,
            
            "dna_helix.txt": """
            DNA Double Helix
            Double helix structure discovered by Watson and Crick in 1953.
            Contains genetic information. Made of nucleotides: A, T, G, C.
            """
        }
        
        # Save text files
        for filename, content in sample_docs.items():
            filepath = sample_dir / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  Created: {filename}")
        
        # Also create simple image descriptions as text files
        image_descriptions = {
            "neural_network_diagram.txt": """
            [IMAGE DESCRIPTION] Neural network diagram showing:
            Input layer (3 nodes) → Hidden layer (5 nodes) → Hidden layer (4 nodes) → Output layer (2 nodes)
            Arrows show connections between layers.
            """,
            "eiffel_tower_image.txt": """
            [IMAGE DESCRIPTION] Eiffel Tower photo:
            Tall iron lattice tower, sunset background,巴黎 skyline visible.
            """,
            "car_engine_diagram.txt": """
            [IMAGE DESCRIPTION] Car engine cross-section diagram:
            Labeled parts: pistons moving up/down, crankshaft rotating, valves opening/closing.
            """
        }
        
        for filename, content in image_descriptions.items():
            filepath = sample_dir / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  Created: {filename}")
        
        # Index the documents
        try:
            self.model.index(
                input_path=str(sample_dir),
                index_name=index_name,
                overwrite=True,
                verbose=0
            )
            self.current_index_name = index_name
            print(f"\n✅ Successfully indexed {len(sample_docs) + len(image_descriptions)} documents as '{index_name}'")
            print(f"   Index location: {self.model.index_root}/{index_name}")
        except Exception as e:
            print(f"⚠️ Indexing failed: {e}")
            print("Will use simulated retrieval mode")
            self.current_index_name = "simulated"
        
        return self.current_index_name is not None
    
    def retrieve(self, query: str, k: int = 3) -> dict:
        """Retrieve relevant document pages for a query"""
        start_time = time.time()
        
        # If no index or model failed, use simulated retrieval
        if self.current_index_name is None or self.model is None:
            return self._simulate_retrieval(query, k, start_time)
        
        try:
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
                    'content': getattr(result, 'text', 'No content available')
                })
                similarities.append(result.score)
            
            return {
                'answer': f"Found {len(results)} relevant pages. Top score: {results[0].score:.3f}" if results else "No results found",
                'latency_ms': latency,
                'cost': self.cost_per_query,
                'retrieved_docs': retrieved_docs,
                'similarities': similarities
            }
        except Exception as e:
            print(f"⚠️ Search error: {e}, using simulation")
            return self._simulate_retrieval(query, k, start_time)
    
    def _simulate_retrieval(self, query: str, k: int, start_time: float) -> dict:
        """Fallback simulation when ColPali isn't available"""
        latency = (time.time() - start_time) * 1000
        
        # Simple keyword matching for demo
        query_lower = query.lower()
        relevance_scores = []
        
        keywords = {
            'neural network': 0.95,
            'diagram': 0.9,
            'architecture': 0.85,
            'eiffel tower': 0.92,
            'paris': 0.88,
            'car engine': 0.91,
            'art deco': 0.89,
            'dna': 0.87
        }
        
        for kw, score in keywords.items():
            if kw in query_lower:
                relevance_scores.append(score)
        
        if not relevance_scores:
            relevance_scores = [0.7]
        
        return {
            'answer': f"[Simulated] ColPali would find documents about: {query[:50]}... (relevance: {max(relevance_scores):.3f})",
            'latency_ms': latency,
            'cost': self.cost_per_query,
            'retrieved_docs': [{'doc_id': f"sample_{i}", 'score': s} for i, s in enumerate(relevance_scores[:k])],
            'similarities': relevance_scores[:k]
        }


# ============================================
# COMPLETE ROUTER WITH COL PALI
# ============================================

class CompleteVisualRouter:
    """Full router with ColPali for visual retrieval"""
    
    from utils.cost_tracker import RealCostTracker
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
        
        # Create sample index
        print("\n📚 Setting up document index...")
        self.colpali.create_sample_index()
        
        # Path mapping
        self.path_names = {
            0: "⚡ Parametric (LLM only)",
            1: "📝 Text-Only Retrieval",
            2: "🖼️ Visual-Only (ColPali)",
            3: "🔀 Hybrid (Text + ColPali)"
        }
        
        self.temperature = 0.7
        
        # Simple text retriever for hybrid
        self.text_retriever = SimpleTextRetriever()
        self.cost_tracker = RealCostTracker()
        self.start_time = None
    
    def route_and_retrieve(self, query: str):
        """Route with real cost tracking"""
        
        # Start timing for GPU usage
        import time
        start_gpu_time = time.time()
        
        # Get real GPU metrics before
        gpu_before = self.cost_tracker.get_real_gpu_metrics()
        
        # ... existing routing code ...
        
        # Get GPU metrics after
        gpu_after = self.cost_tracker.get_real_gpu_metrics()
        gpu_time = time.time() - start_gpu_time
        
        # Determine path used
        if decision == 0:
            path_used = 'parametric'
        elif decision == 1:
            path_used = 'text'
        elif decision == 2:
            path_used = 'visual'
        else:
            path_used = 'hybrid'
        
        # Log real cost
        real_cost = self.cost_tracker.log_query_cost(
            query=query,
            path_used=path_used,
            actual_latency_ms=result['retrieval_result']['latency_ms'],
            gpu_time_seconds=gpu_time if path_used in ['visual', 'hybrid'] else None
        )
        
        # Add real cost to result
        result['real_cost'] = real_cost
        result['gpu_metrics'] = {
            'before': gpu_before,
            'after': gpu_after,
            'duration_seconds': gpu_time
        }
        
        return result
    # ========== FIX 4: Cost-Aware Escalation Strategy ==========
    def _needs_hybrid(self, query: str) -> bool:
        """Check if query truly needs both text and visual"""
        import re
        query_lower = query.lower()
        
        # Strong hybrid indicators (MUST have both text explanation AND visual)
        strong_hybrid_patterns = [
            r'explain how.*(diagram|illustration|labeled)',
            r'how does.*(work|function).*diagram',
            r'compare.*with (examples?|diagram)',
            r'labeled (diagram|illustration)',
            r'show me how.*(diagram|illustration)',
            r'process.*(diagram|flowchart)',
            r'explain.*with (diagram|illustration|labeled)',
            r'what does.*look like',  # Visual-only, NOT hybrid
        ]
        
        for pattern in strong_hybrid_patterns:
            if re.search(pattern, query_lower):
                return True
        
        # Check for combination that indicates hybrid
        has_explain_action = any(kw in query_lower for kw in [
            'explain', 'how does', 'how to', 'works', 'compare', 
            'difference between', 'what is the process', 'steps'
        ])
        
        has_visual = any(kw in query_lower for kw in [
            'diagram', 'illustration', 'labeled', 'chart', 
            'flowchart', 'visualization', 'schematic'
        ])
        
        # Special cases
        if 'show me how' in query_lower and 'diagram' not in query_lower:
            return False  # Pure visual
        
        if 'explain' in query_lower and 'diagram' in query_lower:
            return True  # Likely hybrid
        
        return has_explain_action and has_visual
    def _apply_cost_optimization(self, query: str, result: dict) -> dict:
        """Apply cost optimizations after routing decision."""
        
        path_used = result.get('path_used', 'unknown')
        quality = result.get('quality_estimate', 0.7)
        query_lower = query.lower()
        
        # Optimization: Hybrid → Text when explanation can be textual
        if path_used == 'hybrid' and quality >= 0.6:
            # Check if text would be sufficient
            text_result = self.text_retriever.retrieve(query)
            text_quality = self._estimate_quality(text_result, query)
            
            # If text quality is good, use text instead (cheaper)
            if text_quality >= 0.7:
                print(f"   [💰 OPTIMIZATION: Hybrid → Text (saved $0.01)]")
                text_result['quality_estimate'] = text_quality
                text_result['path_used'] = 'text'
                text_result['cost'] = 0.0005
                return text_result
        
        return result
    def route_with_cost_awareness(self, query: str, quality_threshold: float = 0.7):
        """
        Try cheaper paths first, escalate only if quality is insufficient
        This improves cost savings by using expensive paths only when needed
        """
        """Try cheaper paths first, escalate only if quality is insufficient"""
    
    # ========== FORCE HYBRID FOR SPECIFIC PATTERNS ==========
        import re
        query_lower = query.lower()
        
        # Queries that absolutely need both text and visual
        force_hybrid_patterns = [
            r'explain how.*(diagram|illustration)',
            r'how does.*work.*diagram',
            r'how to.*with diagram',
            r'compare.*with examples',
            r'labeled (diagram|anatomy)',
        ]
        
        for pattern in force_hybrid_patterns:
            if re.search(pattern, query_lower):
                print(f"\n{'='*60}")
                print(f"Query: {query}")
                print(f"Router Decision: 🔀 Hybrid (Text + ColPali) [FORCED - PATTERN MATCH]")
                print(f"{'='*60}")
                
                # Execute hybrid retrieval
                text_result = self.text_retriever.retrieve(query)
                visual_result = self.colpali.retrieve(query)
                hybrid_result = self._merge_results(text_result, visual_result)
                hybrid_result['quality_estimate'] = 0.95
                hybrid_result['path_used'] = 'hybrid'
                hybrid_result['cost'] = 0.0105
                
                return self._format_cost_aware_result(query, hybrid_result)
        # ========== END FORCE HYBRID ==========

        import re
        query_lower = query.lower()
        
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"Cost-Aware Routing (threshold={quality_threshold})")
        print(f"{'='*60}")
        
        # Step 1: Try parametric (cheapest: $0.0001)
        print("\n📚 Step 1: Trying Parametric (LLM knowledge)...")
        if self._is_factual_query(query):
            result = self._parametric_retrieval(query)
            result['quality_estimate'] = 0.85
            result['path_used'] = 'parametric'
            
            if result['quality_estimate'] >= quality_threshold:
                print(f"   ✅ Parametric sufficient! Quality: {result['quality_estimate']:.2f}")
                # ========== APPLY COST OPTIMIZATION HERE ==========
                result = self._apply_cost_optimization(query, result)
                # ==================================================
                return self._format_cost_aware_result(query, result)
            else:
                print(f"   ❌ Parametric insufficient, escalating...")
        else:
            print("   ⏭️  Skipping parametric (not a factual query)")
        
        # Step 2: Try text retrieval (cheap: $0.0005)
        print("\n📄 Step 2: Trying Text-Only Retrieval...")
        text_result = self.text_retriever.retrieve(query)
        text_result['quality_estimate'] = self._estimate_quality(text_result, query)
        text_result['path_used'] = 'text'
        
        if text_result['quality_estimate'] >= quality_threshold:
            print(f"   ✅ Text retrieval sufficient! Quality: {text_result['quality_estimate']:.2f}")
            # ========== APPLY COST OPTIMIZATION HERE ==========
            text_result = self._apply_cost_optimization(query, text_result)
            # ==================================================
            return self._format_cost_aware_result(query, text_result)
        else:
            print(f"   ❌ Text retrieval insufficient (quality: {text_result['quality_estimate']:.2f}), escalating...")
        
        # Step 3: Try visual retrieval (expensive: $0.01)
        print("\n🖼️ Step 3: Trying Visual-Only (ColPali)...")
        visual_result = self.colpali.retrieve(query)
        visual_result['quality_estimate'] = self._estimate_quality(visual_result, query)
        visual_result['path_used'] = 'visual'
        
        # Check if hybrid would be better
        is_hybrid_needed = self._needs_hybrid(query)
        
        if is_hybrid_needed:
            print("\n🔀 Step 4: Using Hybrid (Text + Visual)...")
            text_result = self.text_retriever.retrieve(query)
            visual_result = self.colpali.retrieve(query)
            hybrid_result = self._merge_results(text_result, visual_result)
            hybrid_result['quality_estimate'] = 0.95
            hybrid_result['path_used'] = 'hybrid'
            hybrid_result['cost'] = 0.0105
            
            # ========== APPLY COST OPTIMIZATION HERE ==========
            hybrid_result = self._apply_cost_optimization(query, hybrid_result)
            # ==================================================
            return self._format_cost_aware_result(query, hybrid_result)
        else:
            print(f"   ✅ Visual retrieval sufficient! Quality: {visual_result['quality_estimate']:.2f}")
            # ========== APPLY COST OPTIMIZATION HERE ==========
            visual_result = self._apply_cost_optimization(query, visual_result)
            # ==================================================
            return self._format_cost_aware_result(query, visual_result)
    def _apply_cost_optimization(self, query: str, result: dict) -> dict:
        """
        Apply cost optimizations after routing decision.
        This helps improve cost savings by downgrading expensive paths when cheaper ones would suffice.
        """
        import re
        
        path_used = result.get('path_used', 'unknown')
        quality = result.get('quality_estimate', 0.7)
        query_lower = query.lower()
        
        # Optimization 1: Hybrid → Visual (save $0.0005)
        if path_used == 'hybrid' and quality >= 0.85:
            # Check if visual-only would be sufficient
            visual_keywords = ['diagram', 'illustration', 'chart', 'map', 'graph', 'picture', 'image']
            has_visual = any(kw in query_lower for kw in visual_keywords)
            
            # If query doesn't need text explanation, use visual instead
            needs_text = any(kw in query_lower for kw in ['explain', 'how does', 'how to', 'works', 'compare'])
            
            if has_visual and not needs_text:
                print(f"   [💰 COST OPTIMIZATION: Hybrid → Visual (saved $0.0005)]")
                visual_result = self.colpali.retrieve(query)
                visual_result['quality_estimate'] = quality
                visual_result['path_used'] = 'visual'
                visual_result['cost'] = 0.01
                return visual_result
        
        # Optimization 2: Visual → Text (save $0.0095)
        if path_used == 'visual' and quality >= 0.9:
            # Check if text retrieval would be sufficient
            text_result = self.text_retriever.retrieve(query)
            text_quality = self._estimate_quality(text_result, query)
            
            if text_quality >= 0.85:
                print(f"   [💰 COST OPTIMIZATION: Visual → Text (saved $0.0095)]")
                text_result['quality_estimate'] = text_quality
                text_result['path_used'] = 'text'
                text_result['cost'] = 0.0005
                return text_result
        
        # Optimization 3: Visual → Parametric (save $0.0099)
        if path_used == 'visual' and self._is_factual_query(query):
            print(f"   [💰 COST OPTIMIZATION: Visual → Parametric (saved $0.0099)]")
            parametric_result = self._parametric_retrieval(query)
            parametric_result['quality_estimate'] = 0.85
            parametric_result['path_used'] = 'parametric'
            parametric_result['cost'] = 0.0001
            return parametric_result
        
        # Optimization 4: Lower quality threshold for hybrid queries
        if path_used == 'hybrid' and quality < 0.7:
            # Hybrid with low quality - fall back to visual or text
            visual_result = self.colpali.retrieve(query)
            visual_quality = self._estimate_quality(visual_result, query)
            
            if visual_quality >= 0.6:
                print(f"   [💰 COST OPTIMIZATION: Low-quality Hybrid → Visual (saved $0.0005)]")
                visual_result['quality_estimate'] = visual_quality
                visual_result['path_used'] = 'visual'
                visual_result['cost'] = 0.01
                return visual_result
        
        return result
    
    def _is_factual_query(self, query: str) -> bool:
        """Check if query can be answered from parametric knowledge"""
        import re
        query_lower = query.lower()
        
        # Expanded factual patterns
        factual_patterns = [
            # Basic definitions
            r'^what is (python|sql|api|devops|docker|kubernetes|git|ci/cd)',
            r'^what is (machine learning|deep learning|nlp|computer vision)',
            r'^what is (blockchain|cryptocurrency|cloud computing|serverless)',
            r'^what is (data science|big data|etl|data warehouse)',
            r'^what is (cybersecurity|firewall|vpn|encryption)',
            r'^define',
            r'^calculate',
            r'^formula for',
            
            # Explanations (these are textual, not visual)
            r'^explain (object-oriented|concepts?|principles?)',
            r'^explain (how|why|what)',
            r'^difference between',
            r'^compare (.*) and (.*)',
            
            # Technology concepts
            r'rest api',
            r'container.*virtual',
            r'orchestration',
            r'version control',
            r'ci/cd pipeline',
            r'microservices',
            
            # Business terms
            r'roi', r'ebitda', r'net present value', r'internal rate of return',
            r'balance sheet', r'cash flow', r'profit and loss',
            
            # Science concepts (textual explanations)
            r'photosynthesis process',
            r'newton.*law',
            r'quantum physics',
            r'theory of relativity',
        ]
        
        for pattern in factual_patterns:
            if re.search(pattern, query_lower):
                return True
        
        # Check if query is purely definitional (starts with "what is", "explain", etc.)
        # AND does NOT contain visual keywords
        visual_keywords = ['diagram', 'look like', 'show me', 'picture', 'image', 
                        'illustration', 'chart', 'graph', 'map', 'visual']
        
        definitional_starts = ['what is', 'explain', 'define', 'difference between']
        
        is_definitional = any(query_lower.startswith(start) for start in definitional_starts)
        has_visual_keyword = any(kw in query_lower for kw in visual_keywords)
        
        if is_definitional and not has_visual_keyword:
            return True
        
        return False
    
    def _estimate_quality(self, result: dict, query: str) -> float:
        """Estimate result quality based on relevance scores"""
        similarities = result.get('similarities', [])
        if similarities:
            return max(similarities)
        return 0.6  # Default
    
    def _needs_hybrid(self, query: str) -> bool:
        """Check if query truly needs both text and visual"""
        import re
        query_lower = query.lower()
        
        # Strong hybrid indicators - these MUST use hybrid path
        strong_hybrid_patterns = [
            r'explain how.*(diagram|illustration|labeled)',
            r'how does.*(work|function).*diagram',
            r'compare.*with (examples?|diagram)',
            r'labeled (diagram|illustration|anatomy|structure)',
            r'process.*(diagram|flowchart)',
            r'explain.*with (diagram|illustration|labeled)',
            r'what is the (process|structure|anatomy).*diagram',
            r'show me (the )?(process|structure|anatomy).*diagram',
            r'how to.*with diagram',
            r'works with diagram',
        ]
        
        for pattern in strong_hybrid_patterns:
            if re.search(pattern, query_lower):
                return True
        
        # Check for combination: explanation + visual element
        has_explanation = any(kw in query_lower for kw in [
            'explain', 'how does', 'how to', 'works', 'compare', 
            'difference between', 'what is the process', 'steps', 
            'labeled', 'anatomy', 'structure', 'process'
        ])
        
        has_visual = any(kw in query_lower for kw in [
            'diagram', 'illustration', 'labeled', 'chart', 
            'flowchart', 'visualization', 'schematic', 'anatomy'
        ])
        
        # Special case: queries that start with "explain" AND contain "diagram" are hybrid
        if query_lower.startswith('explain') and has_visual:
            return True
        
        # Special case: "how does X work with diagram" is hybrid
        if 'how does' in query_lower and 'work' in query_lower and has_visual:
            return True
        
        # Special case: "compare X vs Y with examples" is hybrid
        if 'compare' in query_lower and 'with examples' in query_lower:
            return True
        
        # Pure visual queries (not hybrid)
        pure_visual = any(kw in query_lower for kw in [
            'what does.*look like', 'show me (a|the) (picture|image)',
            'photo of', 'image of'
        ])
        
        if pure_visual and not has_explanation:
            return False
        
        # If it has both explanation and visual elements, it's hybrid
        if has_explanation and has_visual:
            return True
        
        return False
    
    def _format_cost_aware_result(self, query: str, result: dict) -> dict:
        """Format result for cost-aware routing"""
        path_names_map = {
            'parametric': "⚡ Parametric (LLM only)",
            'text': "📝 Text-Only Retrieval",
            'visual': "🖼️ Visual-Only (ColPali)",
            'hybrid': "🔀 Hybrid (Text + ColPali)"
        }
        
        path_index_map = {
            'parametric': 0,
            'text': 1,
            'visual': 2,
            'hybrid': 3
        }
        
        path_used = result.get('path_used', 'text')
        
        print(f"\n📄 Final Retrieval Result ({path_used}):")
        print(f"   Answer: {result['answer'][:200]}...")
        print(f"   Latency: {result['latency_ms']:.1f}ms")
        print(f"   Cost: ${result['cost']:.5f}")
        print(f"   Quality Estimate: {result.get('quality_estimate', 0.7):.2f}")
        
        return {
            'query': query,
            'decision': path_index_map[path_used],
            'path_name': path_names_map[path_used],
            'confidence': result.get('quality_estimate', 0.7),
            'retrieval_result': result
        }
    
    # ========== END OF FIX 4 ==========
    
    # ========== MAIN ROUTING METHOD WITH FIXES 2 & 3 ==========
    
    def route_and_retrieve(self, query: str):
        """
        Step 1: Router decides which path to use
        Step 2: Execute actual retrieval based on decision
        """
        
        # ========== FIX 2: Explicit Hybrid Detection ==========
        import re
        query_lower = query.lower()
        
        # Patterns that indicate hybrid queries (need both text + visual)
        hybrid_patterns = [
            r'explain how.*diagram',
            r'how does.*work.*diagram',
            r'explain.*with diagram',
            r'with labeled diagram',
            r'show me how',
            r'process with diagram',
            r'how to.*with diagram',
            r'explain.*illustration',
            r'works with diagram',
            r'compare.*with examples',
        ]
        
        is_hybrid = False
        for pattern in hybrid_patterns:
            if re.search(pattern, query_lower):
                is_hybrid = True
                break
        
        # Also check for combination of explain + visual keywords
        has_explain = any(kw in query_lower for kw in ['explain', 'how does', 'how to', 'works'])
        has_visual = any(kw in query_lower for kw in ['diagram', 'illustration', 'labeled', 'chart', 'map'])
        
        if has_explain and has_visual:
            is_hybrid = True
        
        # Force hybrid decision if patterns match
        if is_hybrid:
            decision = 3  # Hybrid path
            confidence = 0.85
            print(f"\n{'='*60}")
            print(f"Query: {query}")
            print(f"Router Decision: {self.path_names[decision]} (confidence: {confidence:.3f}) [HYBRID PATTERN MATCH]")
            print(f"{'='*60}")
            
            # Execute hybrid retrieval directly
            text_result = self.text_retriever.retrieve(query)
            visual_result = self.colpali.retrieve(query)
            result = self._merge_results(text_result, visual_result)
            
            print(f"\n📄 Retrieval Result:")
            print(f"   Answer: {result['answer'][:200]}...")
            print(f"   Latency: {result['latency_ms']:.1f}ms")
            print(f"   Cost: ${result['cost']:.5f}")
            
            return {
                'query': query,
                'decision': decision,
                'path_name': self.path_names[decision],
                'confidence': confidence,
                'retrieval_result': result
            }
        
        # ========== END OF FIX 2 ==========
        
        # STEP 1: Get router decision
        embedding = self.encoder.encode(query).astype(np.float32)
        emb_tensor = torch.tensor(embedding).unsqueeze(0)
        
        with torch.no_grad():
            outputs = self.router(emb_tensor)
            scaled_logits = outputs['path_logits'] / self.temperature
            probs = torch.softmax(scaled_logits, dim=1)
            decision = torch.argmax(probs, dim=1).item()
            confidence = probs[0, decision].item()
        
        # ========== FIX 3: Clean Up Text vs Visual Confusion ==========
        query_lower = query.lower()
        
        # Force text for purely factual queries (even if they contain visual keywords)
        text_only_patterns = [
            r'what is (computer vision|transformer architecture|machine learning)',
            r'define',
            r'explain (the )?concept',
            r'difference between',
            r'how to calculate',
            r'formula for',
        ]
        
        is_purely_textual = False
        for pattern in text_only_patterns:
            if re.search(pattern, query_lower):
                is_purely_textual = True
                break
        
        # If it's purely textual but router chose visual/hybrid, override
        if is_purely_textual and decision in [2, 3]:
            old_decision = decision
            decision = 1  # Force text-only
            confidence = 0.75
            print(f"   [CORRECTION: '{query[:40]}...' forced to Text-Only (was {self.path_names[old_decision]})]")
        
        # Force visual for queries that clearly need images
        visual_only_patterns = [
            r'what does.*look like',
            r'show me',
            r'diagram of',
            r'picture of',
            r'image of',
            r'illustration of',
            r'chart showing',
            r'graph of',
        ]
        
        is_purely_visual = False
        for pattern in visual_only_patterns:
            if re.search(pattern, query_lower):
                is_purely_visual = True
                break
        
        # If it's purely visual but router chose text, override
        if is_purely_visual and decision in [0, 1]:
            old_decision = decision
            decision = 2  # Force visual
            confidence = 0.85
            print(f"   [CORRECTION: '{query[:40]}...' forced to Visual (was {self.path_names[old_decision]})]")
        
        # ========== END OF FIX 3 ==========
        
        # STEP 2: Execute actual retrieval based on decision
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"Router Decision: {self.path_names[decision]} (confidence: {confidence:.3f})")
        print(f"{'='*60}")
        
        if decision == 0:  # Parametric
            result = self._parametric_retrieval(query)
        elif decision == 1:  # Text-Only
            result = self.text_retriever.retrieve(query)
        elif decision == 2:  # Visual-Only (ColPali)
            result = self.colpali.retrieve(query)
        else:  # Hybrid
            text_result = self.text_retriever.retrieve(query)
            visual_result = self.colpali.retrieve(query)
            result = self._merge_results(text_result, visual_result)
        
        # Display retrieval result
        print(f"\n📄 Retrieval Result:")
        print(f"   Answer: {result['answer'][:200]}...")
        print(f"   Latency: {result['latency_ms']:.1f}ms")
        print(f"   Cost: ${result['cost']:.5f}")
        
        return {
            'query': query,
            'decision': decision,
            'path_name': self.path_names[decision],
            'confidence': confidence,
            'retrieval_result': result
        }
    
    # ========== HELPER METHODS ==========
    
    def _parametric_retrieval(self, query):
        """Simulate LLM parametric knowledge"""
        return {
            'answer': f"[Parametric] Based on my knowledge: {query}",
            'latency_ms': 50,
            'cost': 0.0001,
            'retrieved_docs': []
        }
    
    def _merge_results(self, text_result, visual_result):
        """Merge text and visual results"""
        return {
            'answer': f"[Hybrid] Text: {text_result['answer'][:100]}... | Visual: {visual_result['answer'][:100]}...",
            'latency_ms': text_result['latency_ms'] + visual_result['latency_ms'],
            'cost': text_result['cost'] + visual_result['cost'],
            'retrieved_docs': text_result.get('retrieved_docs', []) + visual_result.get('retrieved_docs', []),
            'similarities': text_result.get('similarities', []) + visual_result.get('similarities', [])
        }

class SimpleTextRetriever:
    def __init__(self):
        self.cost_per_query = 0.0005
        
    def retrieve(self, query: str, k: int = 3) -> dict:
        start_time = time.time()
        query_lower = query.lower()
        
        # Expanded topics with relevance scores
        topics = {
            # Programming
            'python': ('Python is a high-level, interpreted programming language known for its simplicity and readability.', 0.95),
            'object-oriented': ('Object-oriented programming (OOP) is a paradigm based on objects containing data and methods.', 0.95),
            'rest api': ('REST API is an architectural style for designing networked applications using HTTP requests.', 0.95),
            'sql': ('SQL is a domain-specific language for managing relational databases.', 0.95),
            'nosql': ('NoSQL databases are non-tabular databases that store data differently than relational tables.', 0.95),
            'devops': ('DevOps combines development and operations to shorten development lifecycles.', 0.95),
            'docker': ('Docker is a platform for developing, shipping, and running applications in containers.', 0.95),
            'kubernetes': ('Kubernetes is an open-source system for automating container deployment and management.', 0.95),
            'git': ('Git is a distributed version control system for tracking changes in source code.', 0.95),
            'ci/cd': ('CI/CD automates building, testing, and deploying applications.', 0.95),
            'cloud computing': ('Cloud computing delivers computing services over the internet.', 0.95),
            'serverless': ('Serverless computing lets developers build applications without managing servers.', 0.95),
            'microservices': ('Microservices architecture structures an application as independent deployable services.', 0.95),
            'blockchain': ('Blockchain is a distributed ledger technology for secure transactions.', 0.95),
            'machine learning': ('Machine learning enables systems to learn from data without explicit programming.', 0.95),
            'deep learning': ('Deep learning uses neural networks with multiple layers for complex pattern recognition.', 0.95),
            'nlp': ('Natural Language Processing enables computers to understand human language.', 0.95),
            'computer vision': ('Computer vision enables computers to interpret visual information.', 0.95),
            
            # Business
            'roi': ('Return on Investment measures profitability relative to cost.', 0.95),
            'ebitda': ('EBITDA measures operating performance before interest, taxes, depreciation, and amortization.', 0.95),
            'net present value': ('NPV calculates the value of future cash flows in today\'s dollars.', 0.95),
            'balance sheet': ('A balance sheet shows a company\'s assets, liabilities, and equity.', 0.95),
            
            # Science
            'photosynthesis': ('Photosynthesis is the process plants use to convert light into energy.', 0.95),
            'newton': ('Newton\'s laws describe the relationship between motion and forces.', 0.95),
        }
        
        relevant_docs = []
        best_score = 0.6  # Default
        
        for keyword, (content, score) in topics.items():
            if keyword in query_lower:
                relevant_docs.append({'text': content, 'score': score})
                best_score = max(best_score, score)
        
        if not relevant_docs:
            # For general "explain concept" queries
            if any(word in query_lower for word in ['explain', 'what is', 'define']):
                relevant_docs = [{'text': f'Information about {query[:100]}', 'score': 0.75}]
                best_score = 0.75
            else:
                relevant_docs = [{'text': f'General information about {query[:50]}', 'score': 0.6}]
        
        latency = (time.time() - start_time) * 1000
        
        return {
            'answer': f"Found {len(relevant_docs)} relevant documents",
            'latency_ms': latency,
            'cost': self.cost_per_query,
            'retrieved_docs': relevant_docs,
            'similarities': [d['score'] for d in relevant_docs]
        }
# ============================================
# DEMO FUNCTION
# ============================================

def demo_colpali_router():
    """Demonstrate the complete ColPali-enabled router"""
    
    print("="*70)
    print("COL PALI-ENABLED VISUAL ROUTER DEMO")
    print("="*70)
    
    # Initialize the complete router (this will create sample index)
    router = CompleteVisualRouter()
    
    # Test queries that should trigger visual retrieval
    test_queries = [
        "What does a neural network architecture diagram look like?",
        "Show me examples of Art Deco architecture",
        "Explain how a car engine works with diagrams",
        "What does the Eiffel Tower look like?",
        "What is the structure of a DNA double helix?",
    ]
    
    results = []
    print("\n" + "="*70)
    print("RUNNING QUERIES")
    print("="*70)
    
    for query in test_queries:
        result = router.route_and_retrieve(query)
        results.append(result)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for r in results:
        print(f"\n📌 {r['query'][:50]}...")
        print(f"   → {r['path_name']} (conf: {r['confidence']:.3f})")
        print(f"   → Cost: ${r['retrieval_result']['cost']:.5f}")
        print(f"   → Time: {r['retrieval_result']['latency_ms']:.1f}ms")
    
    return results


if __name__ == '__main__':
    demo_colpali_router()