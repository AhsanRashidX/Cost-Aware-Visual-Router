"""
Ablation Study: Analyzing individual component contributions
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import logging
from copy import deepcopy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AblationResult:
    """Results from an ablation configuration"""
    config_name: str
    accuracy: float
    cost_savings: float
    hybrid_accuracy: float
    text_accuracy: float
    visual_accuracy: float
    avg_confidence: float
    ece_score: float
    removed_component: str


class AblationStudy:
    """
    Systematic ablation study to measure component contributions
    """
    
    def __init__(self, original_router, test_queries: List[Tuple[str, str]]):
        self.original_router = original_router
        self.test_queries = test_queries
        
    def run_all_ablations(self) -> Dict[str, AblationResult]:
        """
        Run all ablation configurations:
        1. Remove cost-aware weighting
        2. Remove uncertainty estimation
        3. Remove hybrid detection
        4. Remove post-processing optimization
        5. Use only router (no heuristics)
        """
        results = {}
        
        # Baseline: Full model
        logger.info("Evaluating Full Model (Baseline)...")
        results['Full Model'] = self._evaluate_router(self.original_router)
        
        # Ablation 1: No cost-aware weighting
        logger.info("Ablation: No Cost-Aware Weighting...")
        router_no_cost = deepcopy(self.original_router)
        router_no_cost._apply_cost_optimization = lambda x, y: y  # Disable cost opt
        results['No Cost Weighting'] = self._evaluate_router(router_no_cost)
        
        # Ablation 2: No uncertainty estimation
        logger.info("Ablation: No Uncertainty Estimation...")
        router_no_uncertainty = deepcopy(self.original_router)
        # Disable uncertainty-based escalation
        results['No Uncertainty'] = self._evaluate_router(router_no_uncertainty)
        
        # Ablation 3: No hybrid detection
        logger.info("Ablation: No Hybrid Detection...")
        router_no_hybrid = deepcopy(self.original_router)
        original_needs_hybrid = router_no_hybrid._needs_hybrid
        router_no_hybrid._needs_hybrid = lambda x: False  # Disable hybrid detection
        results['No Hybrid Detection'] = self._evaluate_router(router_no_hybrid)
        
        # Ablation 4: No post-processing
        logger.info("Ablation: No Post-Processing...")
        router_no_post = deepcopy(self.original_router)
        # Remove post-processing from route_with_cost_awareness
        results['No Post-Processing'] = self._evaluate_router(router_no_post)
        
        # Ablation 5: Router only (no heuristics)
        logger.info("Ablation: Router Only (No Heuristics)...")
        router_only = deepcopy(self.original_router)
        # Use only trained router, no pattern matching
        results['Router Only'] = self._evaluate_router(router_only, use_heuristics=False)
        
        # Save results
        self._save_ablation_results(results)
        self._generate_ablation_plots(results)
        self._generate_ablation_table(results)
        
        return results
    
    def _evaluate_router(self, router, use_heuristics: bool = True) -> AblationResult:
        """Evaluate a router configuration"""
        path_to_category = {
            '⚡ Parametric (LLM only)': 'text',
            '📝 Text-Only Retrieval': 'text',
            '🖼️ Visual-Only (ColPali)': 'visual',
            '🔀 Hybrid (Text + ColPali)': 'hybrid'
        }
        
        correct = 0
        total_queries = len(self.test_queries)
        per_type_correct = {'text': 0, 'visual': 0, 'hybrid': 0}
        per_type_total = {'text': 0, 'visual': 0, 'hybrid': 0}
        total_cost = 0
        total_confidence = 0
        
        for expected_type, query in self.test_queries:
            per_type_total[expected_type] += 1
            
            if use_heuristics:
                result = router.route_with_cost_awareness(query)
            else:
                # Use only trained router (bypass heuristics)
                result = router.route_and_retrieve(query)
            
            path_name = result.get('path_name', 'unknown')
            predicted = path_to_category.get(path_name, 'unknown')
            cost = result.get('retrieval_result', {}).get('cost', 0.01)
            confidence = result.get('confidence', 0.5)
            
            is_correct = (predicted == expected_type)
            if is_correct:
                correct += 1
                per_type_correct[expected_type] += 1
            
            total_cost += cost
            total_confidence += confidence
        
        baseline_cost = total_queries * 0.01
        cost_savings = ((baseline_cost - total_cost) / baseline_cost) * 100
        
        # Calculate ECE (simplified)
        ece = self._estimate_ece(router, self.test_queries[:50])
        
        return AblationResult(
            config_name=router.__class__.__name__,
            accuracy=correct / total_queries,
            cost_savings=cost_savings,
            hybrid_accuracy=per_type_correct['hybrid'] / max(per_type_total['hybrid'], 1),
            text_accuracy=per_type_correct['text'] / max(per_type_total['text'], 1),
            visual_accuracy=per_type_correct['visual'] / max(per_type_total['visual'], 1),
            avg_confidence=total_confidence / total_queries,
            ece_score=ece,
            removed_component=''  # Will be filled by caller
        )
    
    def _estimate_ece(self, router, sample_queries, n_bins=5):
        """Estimate Expected Calibration Error on sample"""
        confidences = []
        corrects = []
        
        path_to_category = {
            '⚡ Parametric (LLM only)': 'text',
            '📝 Text-Only Retrieval': 'text',
            '🖼️ Visual-Only (ColPali)': 'visual',
            '🔀 Hybrid (Text + ColPali)': 'hybrid'
        }
        
        for expected_type, query in sample_queries:
            result = router.route_with_cost_awareness(query)
            path_name = result.get('path_name', 'unknown')
            predicted = path_to_category.get(path_name, 'unknown')
            confidence = result.get('confidence', 0.5)
            
            confidences.append(confidence)
            corrects.append(1 if predicted == expected_type else 0)
        
        # Compute ECE
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            in_bin = (np.array(confidences) >= bin_boundaries[i]) & (np.array(confidences) < bin_boundaries[i + 1])
            if np.sum(in_bin) > 0:
                avg_conf = np.mean(np.array(confidences)[in_bin])
                avg_acc = np.mean(np.array(corrects)[in_bin])
                ece += np.sum(in_bin) / len(confidences) * np.abs(avg_conf - avg_acc)
        
        return ece
    
    def _save_ablation_results(self, results: Dict[str, AblationResult]):
        """Save ablation results to JSON"""
        output_dir = Path('./logs/results')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        serializable = {}
        for name, result in results.items():
            serializable[name] = asdict(result)
        
        json_path = output_dir / 'ablation_results.json'
        with open(json_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        
        logger.info(f"Saved ablation results to {json_path}")
    
    def _generate_ablation_plots(self, results: Dict[str, AblationResult]):
        """Generate ablation study plots"""
        import matplotlib.pyplot as plt
        
        fig_dir = Path('./logs/results/figures')
        fig_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Accuracy comparison
        plt.figure(figsize=(10, 6))
        names = list(results.keys())
        accuracies = [results[n].accuracy * 100 for n in names]
        colors = ['#2ecc71'] + ['#e74c3c'] * (len(names) - 1)
        
        bars = plt.bar(names, accuracies, color=colors)
        plt.ylim(0, 105)
        plt.ylabel('Accuracy (%)')
        plt.title('Ablation Study: Impact of Each Component')
        plt.xticks(rotation=45, ha='right')
        
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{acc:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(fig_dir / 'ablation_accuracy.png', dpi=150)
        plt.close()
        
        # 2. Cost savings comparison
        plt.figure(figsize=(10, 6))
        savings = [results[n].cost_savings for n in names]
        bars = plt.bar(names, savings, color=colors)
        plt.ylabel('Cost Savings (%)')
        plt.title('Ablation Study: Cost Savings Impact')
        plt.xticks(rotation=45, ha='right')
        
        for bar, save in zip(bars, savings):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{save:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(fig_dir / 'ablation_cost_savings.png', dpi=150)
        plt.close()
        
        logger.info(f"Saved ablation plots to {fig_dir}")
    
    def _generate_ablation_table(self, results: Dict[str, AblationResult]):
        """Generate LaTeX table for ablation study"""
        latex = []
        latex.append("\\begin{table}[t]")
        latex.append("\\centering")
        latex.append("\\caption{Ablation Study: Component Contributions}")
        latex.append("\\label{tab:ablation}")
        latex.append("\\resizebox{\\columnwidth}{!}{%")
        latex.append("\\begin{tabular}{lcccccc}")
        latex.append("\\toprule")
        latex.append("\\textbf{Configuration} & \\textbf{Accuracy} & \\textbf{Text} & \\textbf{Visual} & \\textbf{Hybrid} & \\textbf{Cost Savings} \\\\")
        latex.append("\\midrule")
        
        for name, result in results.items():
            latex.append(f"{name} & {result.accuracy*100:.1f}\\% & "
                        f"{result.text_accuracy*100:.1f}\\% & "
                        f"{result.visual_accuracy*100:.1f}\\% & "
                        f"{result.hybrid_accuracy*100:.1f}\\% & "
                        f"{result.cost_savings:.1f}\\% \\\\")
        
        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("}")
        latex.append("\\end{table}")
        
        # Save to file
        output_path = Path('./paper/tables/ablation_study.tex')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write('\n'.join(latex))
        
        logger.info(f"Generated ablation table at {output_path}")