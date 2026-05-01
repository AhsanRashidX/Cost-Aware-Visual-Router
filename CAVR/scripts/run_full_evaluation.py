#!/usr/bin/env python
"""
Full Evaluation Pipeline for Cost-Aware Visual Router
Runs all benchmarks, baselines, ablations, and generates paper-ready outputs
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from colpali_router_demo import CompleteVisualRouter

from benchmarks.docvqa_loader import DocVQADataset, InfoVQADataset, ArxivQADataset, load_all_benchmarks
from evaluation.router_evaluator import RouterEvaluator
from evaluation.baseline_comparator import BaselineComparator
from evaluation.ablation_study import AblationStudy

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_full_evaluation():
    """Run complete evaluation pipeline"""
    
    logger.info("=" * 70)
    logger.info("FULL EVALUATION PIPELINE")
    logger.info("=" * 70)
    
    # Create output directories
    output_dirs = [
        './logs/results/figures',
        './logs/results/tables',
        './paper/tables',
        './paper/figures'
    ]
    for d in output_dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    
    # Initialize router
    logger.info("\n[1/6] Initializing Router...")
    router = CompleteVisualRouter()
    
    # Load test queries (500+ samples)
    logger.info("\n[2/6] Loading Test Queries...")
    test_queries = load_large_test_set()
    logger.info(f"Loaded {len(test_queries)} test queries")
    
    # Run benchmark evaluations
    logger.info("\n[3/6] Running Benchmark Evaluations...")
    benchmark_results = run_benchmarks(router, test_queries)
    
    # Run baseline comparisons
    logger.info("\n[4/6] Running Baseline Comparisons...")
    baseline_results = run_baseline_comparisons(router, test_queries)
    
    # Run ablation study
    logger.info("\n[5/6] Running Ablation Study...")
    ablation_results = run_ablation_study(router, test_queries)
    
    # Generate paper outputs
    logger.info("\n[6/6] Generating Paper Outputs...")
    generate_paper_outputs(benchmark_results, baseline_results, ablation_results)
    
    # Print summary
    print_summary(benchmark_results, baseline_results, ablation_results)
    
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION COMPLETE!")
    logger.info("Results saved to ./logs/results/")
    logger.info("Paper outputs saved to ./paper/")
    logger.info("=" * 70)


def load_large_test_set(min_samples: int = 600) -> list:
    """Load or create large test set with at least min_samples"""
    
    # Try to load from existing file
    cache_path = Path('./logs/test_queries.json')
    
    if cache_path.exists():
        with open(cache_path, 'r') as f:
            data = json.load(f)
            if len(data) >= min_samples:
                logger.info(f"Loaded {len(data)} queries from cache")
                return [(item['type'], item['query']) for item in data]
    
    # Create new test set
    logger.info(f"Creating new test set with {min_samples} queries...")
    test_queries = []
    
    # Text queries (30% of total)
    text_samples = [
        ("text", f"What is {topic}?") for topic in [
            "Python programming", "SQL database", "REST API", "machine learning",
            "cloud computing", "blockchain", "cybersecurity", "data science",
            "artificial intelligence", "natural language processing", "computer vision",
            "deep learning", "reinforcement learning", "transformer architecture",
            "neural network", "big data", "ETL process", "data warehousing",
            "business intelligence", "DevOps", "Docker", "Kubernetes", "Git",
            "CI/CD pipeline", "serverless computing", "microservices",
            "object-oriented programming", "functional programming", "algorithm",
            "data structure", "operating system", "computer network", "database"
        ]
    ]
    
    # Visual queries (40% of total)
    visual_samples = [
        ("visual", f"What does a {topic} diagram look like?") for topic in [
            "neural network architecture", "DNA double helix", "photosynthesis process",
            "human heart anatomy", "car engine", "water cycle", "rock cycle",
            "cell structure", "solar system", "earth layers", "volcano eruption",
            "eye anatomy", "brain lobes", "neuron synapse", "skeletal system",
            "muscular system", "respiratory system", "digestive system", "nervous system",
            "art deco architecture", "gothic cathedral", "modernist building",
            "baroque architecture", "renaissance architecture", "islamic geometric patterns",
            "japanese temple", "chinese pagoda", "egyptian pyramid", "greek column",
            "color wheel", "perspective drawing", "network topology", "cloud architecture",
            "soccer field", "basketball court", "tennis court", "baseball field",
            "golf swing", "yoga pose", "sewing pattern", "fabric weave", "shoe construction"
        ]
    ]
    
    # Hybrid queries (30% of total)
    hybrid_samples = [
        ("hybrid", f"Explain how {topic} works with diagram") for topic in [
            "a car engine", "solar panel", "rocket propulsion", "blockchain",
            "neural network", "DNA replication", "photosynthesis", "immune response",
            "electrical circuit", "water cycle", "supply chain", "rocket propulsion",
            "solar panel", "car engine", "neural network", "blockchain", "water cycle",
            "immune response", "photosynthesis", "DNA replication", "electrical circuit",
            "supply chain", "rocket propulsion", "solar panel", "car engine"
        ]
    ]
    
    # Add text queries
    for i in range(min_samples // 3):
        if i < len(text_samples):
            test_queries.append(text_samples[i % len(text_samples)])
        else:
            test_queries.append(("text", f"What is concept_{i % 50}?"))
    
    # Add visual queries
    for i in range(min_samples // 3 * 4 // 10):  # ~40%
        if i < len(visual_samples):
            test_queries.append(visual_samples[i % len(visual_samples)])
        else:
            test_queries.append(("visual", f"What does diagram_{i % 50} look like?"))
    
    # Add hybrid queries
    for i in range(min_samples // 3):
        if i < len(hybrid_samples):
            test_queries.append(hybrid_samples[i % len(hybrid_samples)])
        else:
            test_queries.append(("hybrid", f"Explain how concept_{i % 50} works with diagram"))
    
    # Shuffle
    np.random.seed(42)
    np.random.shuffle(test_queries)
    
    # Save to cache
    cache_data = [{'type': t, 'query': q} for t, q in test_queries]
    with open(cache_path, 'w') as f:
        json.dump(cache_data, f, indent=2)
    
    logger.info(f"Created {len(test_queries)} test queries")
    return test_queries


def run_benchmarks(router, test_queries):
    """Run benchmark evaluations"""
    from benchmarks.docvqa_loader import DocVQADataset, InfoVQADataset, ArxivQADataset
    
    evaluator = RouterEvaluator(router, output_dir='./logs/results')
    results = {}
    
    # Run on custom test set
    logger.info("  - Evaluating on Custom Test Set...")
    results['Custom'] = evaluator.evaluate_on_benchmark(
        CustomDataset(test_queries), 'Custom', max_queries=500
    )
    
    # Run on DocVQA-style data
    logger.info("  - Evaluating on DocVQA-style data...")
    docvqa = DocVQADataset('./data/docvqa', split='val', max_samples=300)
    results['DocVQA'] = evaluator.evaluate_on_benchmark(docvqa, 'DocVQA', max_queries=300)
    
    # Run on InfoVQA-style data
    logger.info("  - Evaluating on InfoVQA-style data...")
    infovqa = InfoVQADataset('./data/infovqa', split='val', max_samples=300)
    results['InfoVQA'] = evaluator.evaluate_on_benchmark(infovqa, 'InfoVQA', max_queries=300)
    
    # Run on ArxivQA-style data
    logger.info("  - Evaluating on ArxivQA-style data...")
    arxivqa = ArxivQADataset('./data/arxivqa', split='val', max_samples=200)
    results['ArxivQA'] = evaluator.evaluate_on_benchmark(arxivqa, 'ArxivQA', max_queries=200)
    
    return results


def run_baseline_comparisons(router, test_queries):
    """Run baseline comparisons"""
    comparator = BaselineComparator(router)
    results = comparator.evaluate_all(test_queries, max_queries=500)
    return results


def run_ablation_study(router, test_queries):
    """Run ablation study"""
    ablation = AblationStudy(router, test_queries[:200])  # Use 200 queries for ablation
    results = ablation.run_all_ablations()
    return results


def generate_paper_outputs(benchmark_results, baseline_results, ablation_results):
    """Generate all paper-ready outputs"""
    
    # Generate main results table
    generate_main_results_table(benchmark_results, baseline_results)
    
    # Generate per-benchmark detailed tables
    generate_benchmark_tables(benchmark_results)
    
    # Generate calibration plots
    generate_calibration_plots(benchmark_results)
    
    # Generate summary statistics JSON
    generate_summary_json(benchmark_results, baseline_results, ablation_results)


def generate_main_results_table(benchmark_results, baseline_results):
    """Generate main results table for paper"""
    
    latex = []
    latex.append("\\begin{table*}[t]")
    latex.append("\\centering")
    latex.append("\\caption{Main Results Across Benchmarks}")
    latex.append("\\label{tab:main_results}")
    latex.append("\\resizebox{\\textwidth}{!}{%")
    latex.append("\\begin{tabular}{lcccccc}")
    latex.append("\\toprule")
    latex.append("\\textbf{Benchmark} & \\textbf{Accuracy} & \\textbf{Text} & \\textbf{Visual} & \\textbf{Hybrid} & \\textbf{Cost Savings} \\\\")
    latex.append("\\midrule")
    
    for name, metrics in benchmark_results.items():
        if metrics:
            latex.append(f"{name} & {metrics.accuracy*100:.1f}\\% & "
                        f"{metrics.per_type_accuracy.get('text', 0)*100:.1f}\\% & "
                        f"{metrics.per_type_accuracy.get('visual', 0)*100:.1f}\\% & "
                        f"{metrics.per_type_accuracy.get('hybrid', 0)*100:.1f}\\% & "
                        f"{metrics.cost_savings:.1f}\\% \\\\")
    
    latex.append("\\midrule")
    latex.append("\\multicolumn{7}{c}{\\textbf{Baseline Comparison}} \\\\")
    latex.append("\\midrule")
    
    for name, result in baseline_results.items():
        latex.append(f"{name} & {result.accuracy*100:.1f}\\% & "
                    f"{result.text_accuracy*100:.1f}\\% & "
                    f"{result.visual_accuracy*100:.1f}\\% & "
                    f"{result.hybrid_accuracy*100:.1f}\\% & - \\\\")
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("}")
    latex.append("\\end{table*}")
    
    # Save to file
    output_path = Path('./paper/tables/main_results.tex')
    with open(output_path, 'w') as f:
        f.write('\n'.join(latex))
    
    logger.info(f"Generated main results table at {output_path}")


def generate_benchmark_tables(benchmark_results):
    """Generate detailed per-benchmark tables"""
    
    for name, metrics in benchmark_results.items():
        if not metrics:
            continue
        
        latex = []
        latex.append("\\begin{table}[t]")
        latex.append("\\centering")
        latex.append(f"\\caption{{Detailed Results on {name}}}")
        latex.append(f"\\label{{tab:{name.lower()}_details}}")
        latex.append("\\begin{tabular}{lcc}")
        latex.append("\\toprule")
        latex.append("\\textbf{Metric} & \\textbf{Value} & \\textbf{95\\% CI} \\\\")
        latex.append("\\midrule")
        
        latex.append(f"Accuracy & {metrics.accuracy*100:.1f}\\% & "
                    f"[{metrics.accuracy_ci_lower*100:.1f}\\% - {metrics.accuracy_ci_upper*100:.1f}\\%] \\\\")
        latex.append(f"Cost Savings & {metrics.cost_savings:.1f}\\% & "
                    f"[{metrics.cost_savings_ci[0]:.1f}\\% - {metrics.cost_savings_ci[1]:.1f}\\%] \\\\")
        latex.append(f"Avg Confidence & {metrics.avg_confidence:.3f} & - \\\\")
        latex.append(f"ECE Score & {metrics.ece_score:.3f} & - \\\\")
        latex.append(f"Avg Latency & {metrics.avg_latency_ms:.1f}ms & - \\\\")
        
        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")
        
        output_path = Path(f'./paper/tables/{name.lower()}_details.tex')
        with open(output_path, 'w') as f:
            f.write('\n'.join(latex))
    
    logger.info("Generated per-benchmark detail tables")


def generate_calibration_plots(benchmark_results):
    """Generate calibration reliability plots"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, (name, metrics) in enumerate(benchmark_results.items()):
        if idx >= 4 or not metrics:
            continue
        
        ax = axes[idx]
        curve = metrics.reliability_curve
        if curve:
            confs, accs = zip(*curve)
            ax.plot(confs, accs, 'o-', label='Our Router', linewidth=2, markersize=8)
            ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=1.5)
            ax.set_xlabel('Confidence', fontsize=11)
            ax.set_ylabel('Accuracy', fontsize=11)
            ax.set_title(f'{name} (ECE={metrics.ece_score:.3f})', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.suptitle('Calibration Reliability Curves', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('./paper/figures/calibration_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info("Generated calibration plots")


def generate_summary_json(benchmark_results, baseline_results, ablation_results):
    """Generate summary statistics JSON for reference"""
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'benchmarks': {},
        'baselines': {},
        'ablation': {}
    }
    
    for name, metrics in benchmark_results.items():
        if metrics:
            summary['benchmarks'][name] = {
                'accuracy': metrics.accuracy,
                'accuracy_ci': [metrics.accuracy_ci_lower, metrics.accuracy_ci_upper],
                'cost_savings': metrics.cost_savings,
                'cost_savings_ci': metrics.cost_savings_ci,
                'ece_score': metrics.ece_score,
                'per_type_accuracy': metrics.per_type_accuracy,
                'avg_latency_ms': metrics.avg_latency_ms
            }
    
    for name, result in baseline_results.items():
        summary['baselines'][name] = {
            'accuracy': result.accuracy,
            'cost': result.cost,
            'avg_latency_ms': result.avg_latency_ms,
            'text_accuracy': result.text_accuracy,
            'visual_accuracy': result.visual_accuracy,
            'hybrid_accuracy': result.hybrid_accuracy
        }
    
    for name, result in ablation_results.items():
        summary['ablation'][name] = {
            'accuracy': result.accuracy,
            'cost_savings': result.cost_savings,
            'text_accuracy': result.text_accuracy,
            'visual_accuracy': result.visual_accuracy,
            'hybrid_accuracy': result.hybrid_accuracy,
            'ece_score': result.ece_score
        }
    
    with open('./logs/results/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("Generated summary JSON")


def print_summary(benchmark_results, baseline_results, ablation_results):
    """Print evaluation summary to console"""
    
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    
    print("\n📊 Benchmark Results:")
    print("-" * 50)
    for name, metrics in benchmark_results.items():
        if metrics:
            print(f"  {name}:")
            print(f"    Accuracy: {metrics.accuracy*100:.1f}% "
                  f"(95% CI: [{metrics.accuracy_ci_lower*100:.1f}%, {metrics.accuracy_ci_upper*100:.1f}%])")
            print(f"    Cost Savings: {metrics.cost_savings:.1f}%")
            print(f"    ECE: {metrics.ece_score:.3f}")
            print(f"    Text/Visual/Hybrid: {metrics.per_type_accuracy.get('text', 0)*100:.1f}% / "
                  f"{metrics.per_type_accuracy.get('visual', 0)*100:.1f}% / "
                  f"{metrics.per_type_accuracy.get('hybrid', 0)*100:.1f}%")
    
    print("\n📊 Baseline Comparison:")
    print("-" * 50)
    for name, result in baseline_results.items():
        print(f"  {name}: {result.accuracy*100:.1f}% (Cost: ${result.cost:.4f})")
    
    print("\n📊 Ablation Study:")
    print("-" * 50)
    for name, result in ablation_results.items():
        print(f"  {name}: {result.accuracy*100:.1f}% (Cost Savings: {result.cost_savings:.1f}%)")
    
    # Highlight achievement
    print("\n" + "=" * 70)
    print("🎯 TARGET ACHIEVEMENT")
    print("=" * 70)
    
    avg_accuracy = np.mean([m.accuracy for m in benchmark_results.values() if m])
    avg_savings = np.mean([m.cost_savings for m in benchmark_results.values() if m])
    
    if avg_accuracy >= 85:
        print(f"  ✅ Accuracy target (≥85%): {avg_accuracy*100:.1f}% ACHIEVED")
    else:
        print(f"  ❌ Accuracy target (≥85%): {avg_accuracy*100:.1f}% NOT ACHIEVED")
    
    if avg_savings >= 40:
        print(f"  ✅ Cost savings target (≥40%): {avg_savings:.1f}% ACHIEVED")
    else:
        print(f"  ❌ Cost savings target (≥40%): {avg_savings:.1f}% NOT ACHIEVED")


class CustomDataset:
    """Wrapper for custom test queries to match Dataset interface"""
    def __init__(self, test_queries):
        self.test_queries = test_queries
    
    def __len__(self):
        return len(self.test_queries)
    
    def __getitem__(self, idx):
        exp_type, query = self.test_queries[idx]
        return {
            'question_id': idx,
            'question': query,
            'expected_type': exp_type,
            'answers': [],
            'doc_id': f'custom_{idx}',
            'split': 'val'
        }


if __name__ == '__main__':
    run_full_evaluation()