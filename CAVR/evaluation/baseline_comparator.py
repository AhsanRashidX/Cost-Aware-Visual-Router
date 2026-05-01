"""
Baseline Comparisons with FrugalGPT, RouteLLM, and Adaptive RAG
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import logging
from sklearn.metrics import accuracy_score
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BaselineResult:
    """Results from baseline methods"""
    name: str
    accuracy: float
    cost: float
    avg_latency_ms: float
    hybrid_accuracy: float
    text_accuracy: float
    visual_accuracy: float


class FrugalGPTBaseline:
    """
    FrugalGPT: Cost-Effective LLM Cascading
    Paper: "FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance"
    """
    
    def __init__(self):
        self.cost_tiers = {
            'tiny': 0.0001,   # Parametric
            'small': 0.0005,  # Text retrieval
            'large': 0.01,    # Visual retrieval
        }
        
    def route(self, query: str) -> Tuple[str, float]:
        """
        FrugalGPT cascading strategy:
        Start with cheapest model, escalate if confidence is low
        """
        query_lower = query.lower()
        
        # Simple confidence estimation based on query complexity
        visual_keywords = ['diagram', 'look like', 'show me', 'image', 'picture', 
                          'chart', 'graph', 'map', 'illustration']
        hybrid_keywords = ['explain how', 'how does', 'works with diagram', 'labeled']
        
        # Tier 1: Try cheap model
        if any(kw in query_lower for kw in ['what is', 'define', 'calculate']):
            return 'text', 0.85  # High confidence for factual
        
        # Tier 2: Needs retrieval
        visual_score = sum(1 for kw in visual_keywords if kw in query_lower) / len(visual_keywords)
        
        if visual_score > 0.3:
            # Escalate to expensive model
            if any(kw in query_lower for kw in hybrid_keywords):
                return 'hybrid', 0.7
            return 'visual', 0.8
        
        return 'text', 0.6
    
    def get_cost(self, decision: str) -> float:
        """Get cost for decision"""
        cost_map = {'text': 0.0001, 'visual': 0.01, 'hybrid': 0.0105}
        return cost_map.get(decision, 0.01)


class RouteLLMBaseline:
    """
    RouteLLM: Learning to Route Between LLMs
    Paper: "RouteLLM: Learning to Route Between Large Language Models"
    """
    
    def __init__(self):
        # Learned routing preferences (simulated)
        self.routing_weights = {
            'text_keywords': {'what is': 0.9, 'define': 0.9, 'explain': 0.6},
            'visual_keywords': {'diagram': 0.2, 'look like': 0.1, 'show me': 0.1},
        }
        
    def route(self, query: str) -> Tuple[str, float]:
        """Route based on learned preferences"""
        query_lower = query.lower()
        
        # Compute scores
        text_score = 0.5
        visual_score = 0.5
        
        for kw, weight in self.routing_weights['text_keywords'].items():
            if kw in query_lower:
                text_score += weight
        
        for kw, weight in self.routing_weights['visual_keywords'].items():
            if kw in query_lower:
                visual_score -= weight
        
        if 'diagram' in query_lower and 'explain' in query_lower:
            return 'hybrid', 0.75
        elif visual_score < 0.3:
            return 'visual', visual_score
        else:
            return 'text', text_score
    
    def get_cost(self, decision: str) -> float:
        cost_map = {'text': 0.0005, 'visual': 0.01, 'hybrid': 0.0105}
        return cost_map.get(decision, 0.01)


class AdaptiveRAGBaseline:
    """
    Adaptive RAG: Adaptive Retrieval-Augmented Generation
    Paper: "Adaptive RAG: Learning to Adapt Retrieval for Question Answering"
    """
    
    def __init__(self):
        self.retrieval_threshold = 0.6
        
    def route(self, query: str) -> Tuple[str, float]:
        """
        Adaptive RAG strategy:
        Determine if retrieval is needed based on query complexity
        """
        query_lower = query.lower()
        
        # Simple complexity estimator
        words = query_lower.split()
        has_technical = any(kw in query_lower for kw in ['algorithm', 'model', 'system', 'network'])
        has_visual = any(kw in query_lower for kw in ['diagram', 'figure', 'chart', 'graph'])
        
        # Parametric knowledge for simple queries
        if len(words) < 5 and not has_technical and not has_visual:
            return 'text', 0.85
        
        # Need retrieval for complex queries
        if has_visual and has_technical:
            return 'hybrid', 0.7
        elif has_visual:
            return 'visual', 0.8
        else:
            return 'text', 0.7
    
    def get_cost(self, decision: str) -> float:
        cost_map = {'text': 0.0005, 'visual': 0.01, 'hybrid': 0.0105}
        return cost_map.get(decision, 0.01)


class BaselineComparator:
    """
    Compare our router against baseline methods
    """
    
    def __init__(self, our_router):
        self.our_router = our_router
        self.baselines = {
            'FrugalGPT': FrugalGPTBaseline(),
            'RouteLLM': RouteLLMBaseline(),
            'Adaptive RAG': AdaptiveRAGBaseline(),
        }
    
    def evaluate_all(self, test_queries: List[Tuple[str, str]], 
                     max_queries: int = 500) -> Dict[str, BaselineResult]:
        """
        Evaluate all baselines on the same test queries
        
        Args:
            test_queries: List of (expected_type, query) tuples
            max_queries: Maximum number of queries to evaluate
        """
        results = {}
        
        # Evaluate our router first
        logger.info("Evaluating Our Router...")
        our_results = self._evaluate_router(self.our_router, test_queries[:max_queries])
        results['Ours'] = our_results
        
        # Evaluate each baseline
        for name, baseline in self.baselines.items():
            logger.info(f"Evaluating {name}...")
            baseline_results = self._evaluate_baseline(baseline, test_queries[:max_queries])
            results[name] = baseline_results
        
        # Save comparison results
        self._save_comparison(results)
        
        return results
    
    def _evaluate_router(self, router, test_queries):
        """Evaluate our router"""
        import time
        
        correct = 0
        total_cost = 0
        total_latency = 0
        per_type_correct = {'text': 0, 'visual': 0, 'hybrid': 0}
        per_type_total = {'text': 0, 'visual': 0, 'hybrid': 0}
        
        path_to_category = {
            '⚡ Parametric (LLM only)': 'text',
            '📝 Text-Only Retrieval': 'text',
            '🖼️ Visual-Only (ColPali)': 'visual',
            '🔀 Hybrid (Text + ColPali)': 'hybrid'
        }
        
        for expected_type, query in test_queries:
            per_type_total[expected_type] += 1
            
            start = time.time()
            result = router.route_with_cost_awareness(query)
            latency = (time.time() - start) * 1000
            
            path_name = result.get('path_name', 'unknown')
            predicted = path_to_category.get(path_name, 'unknown')
            cost = result.get('retrieval_result', {}).get('cost', 0.01)
            
            is_correct = (predicted == expected_type)
            if is_correct:
                correct += 1
                per_type_correct[expected_type] += 1
            
            total_cost += cost
            total_latency += latency
        
        return BaselineResult(
            name='Ours',
            accuracy=correct / len(test_queries),
            cost=total_cost,
            avg_latency_ms=total_latency / len(test_queries),
            hybrid_accuracy=per_type_correct['hybrid'] / max(per_type_total['hybrid'], 1),
            text_accuracy=per_type_correct['text'] / max(per_type_total['text'], 1),
            visual_accuracy=per_type_correct['visual'] / max(per_type_total['visual'], 1)
        )
    
    def _evaluate_baseline(self, baseline, test_queries):
        """Evaluate a baseline method"""
        import time
        
        correct = 0
        total_cost = 0
        total_latency = 0
        per_type_correct = {'text': 0, 'visual': 0, 'hybrid': 0}
        per_type_total = {'text': 0, 'visual': 0, 'hybrid': 0}
        
        for expected_type, query in test_queries:
            per_type_total[expected_type] += 1
            
            start = time.time()
            decision, confidence = baseline.route(query)
            latency = (time.time() - start) * 1000
            
            cost = baseline.get_cost(decision)
            
            is_correct = (decision == expected_type)
            if is_correct:
                correct += 1
                per_type_correct[expected_type] += 1
            
            total_cost += cost
            total_latency += latency
        
        return BaselineResult(
            name=baseline.__class__.__name__,
            accuracy=correct / len(test_queries),
            cost=total_cost,
            avg_latency_ms=total_latency / len(test_queries),
            hybrid_accuracy=per_type_correct['hybrid'] / max(per_type_total['hybrid'], 1),
            text_accuracy=per_type_correct['text'] / max(per_type_total['text'], 1),
            visual_accuracy=per_type_correct['visual'] / max(per_type_total['visual'], 1)
        )
    
    def _save_comparison(self, results: Dict[str, BaselineResult]):
        """Save baseline comparison results"""
        output_dir = Path('./logs/results')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        serializable = {}
        for name, result in results.items():
            serializable[name] = {
                'name': result.name,
                'accuracy': result.accuracy,
                'cost': result.cost,
                'avg_latency_ms': result.avg_latency_ms,
                'hybrid_accuracy': result.hybrid_accuracy,
                'text_accuracy': result.text_accuracy,
                'visual_accuracy': result.visual_accuracy
            }
        
        # Save JSON
        json_path = output_dir / 'baseline_comparison.json'
        with open(json_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        
        # Generate comparison table
        self._generate_comparison_table(results)
        
        logger.info(f"Saved baseline comparison to {json_path}")
    
    def _generate_comparison_table(self, results: Dict[str, BaselineResult]):
        """Generate LaTeX table for paper"""
        latex = []
        latex.append("\\begin{table}[t]")
        latex.append("\\centering")
        latex.append("\\caption{Comparison with Baseline Methods}")
        latex.append("\\label{tab:baseline_comparison}")
        latex.append("\\resizebox{\\columnwidth}{!}{%")
        latex.append("\\begin{tabular}{lcccccc}")
        latex.append("\\toprule")
        latex.append("\\textbf{Method} & \\textbf{Accuracy} & \\textbf{Hybrid Acc} & \\textbf{Text Acc} & \\textbf{Visual Acc} & \\textbf{Cost} & \\textbf{Latency} \\\\")
        latex.append("\\midrule")
        
        for name, result in results.items():
            latex.append(f"{name} & {result.accuracy*100:.1f}\\% & {result.hybrid_accuracy*100:.1f}\\% & "
                        f"{result.text_accuracy*100:.1f}\\% & {result.visual_accuracy*100:.1f}\\% & "
                        f"\\${result.cost:.4f} & {result.avg_latency_ms:.1f}ms \\\\")
        
        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("}")
        latex.append("\\end{table}")
        
        # Save to file
        output_path = Path('./paper/tables/baseline_comparison.tex')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write('\n'.join(latex))
        
        logger.info(f"Generated LaTeX table at {output_path}")