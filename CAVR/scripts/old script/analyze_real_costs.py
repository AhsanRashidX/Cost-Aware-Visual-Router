"""
Analyze real cost measurements from API logs
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def analyze_costs():
    """Analyze real cost data"""
    
    log_file = Path('./logs/real_costs.json')
    
    if not log_file.exists():
        print("❌ No cost logs found. Run router with cost tracking first.")
        return
    
    with open(log_file, 'r') as f:
        sessions = json.load(f)
    
    print("="*60)
    print("REAL COST ANALYSIS")
    print("="*60)
    
    total_cost = 0
    total_queries = 0
    all_costs = []
    
    for session in sessions:
        total_cost += session['total_cost']
        total_queries += session['summary']['total_queries']
        
        # Extract query costs
        for call in session['api_calls']:
            if 'components' in call:  # Query record
                all_costs.append(call['total_cost'])
    
    print(f"\n💰 Cost Summary:")
    print(f"   Total Sessions: {len(sessions)}")
    print(f"   Total Queries: {total_queries}")
    print(f"   Total Cost: ${total_cost:.4f}")
    print(f"   Avg Cost/Query: ${total_cost/total_queries:.4f}")
    
    print(f"\n📊 Cost Distribution:")
    print(f"   Min Cost: ${min(all_costs):.4f}")
    print(f"   Max Cost: ${max(all_costs):.4f}")
    print(f"   Median Cost: ${np.median(all_costs):.4f}")
    print(f"   Std Dev: ${np.std(all_costs):.4f}")
    
    # Plot cost distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histogram
    axes[0].hist(all_costs, bins=20, edgecolor='black')
    axes[0].set_xlabel('Cost per Query ($)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Query Costs')
    
    # Box plot by cost range
    cost_ranges = {
        'Low (<$0.001)': [c for c in all_costs if c < 0.001],
        'Medium ($0.001-0.005)': [c for c in all_costs if 0.001 <= c < 0.005],
        'High ($0.005-0.01)': [c for c in all_costs if 0.005 <= c < 0.01],
        'Very High (>$0.01)': [c for c in all_costs if c >= 0.01]
    }
    
    labels = list(cost_ranges.keys())
    counts = [len(v) for v in cost_ranges.values()]
    colors = ['#2ecc71', '#f39c12', '#e74c3c', '#c0392b']
    
    axes[1].bar(labels, counts, color=colors)
    axes[1].set_ylabel('Number of Queries')
    axes[1].set_title('Queries by Cost Range')
    axes[1].set_xticklabels(labels, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('./paper/figures/real_cost_distribution.png', dpi=150)
    plt.close()
    
    print(f"\n📈 Cost Categories:")
    for label, count in cost_ranges.items():
        pct = count / len(all_costs) * 100
        print(f"   {label}: {count} queries ({pct:.1f}%)")
    
    # Generate paper table
    generate_cost_table(total_cost, total_queries, all_costs)

def generate_cost_table(total_cost, total_queries, all_costs):
    """Generate LaTeX table for paper"""
    
    latex = []
    latex.append("\\begin{table}[t]")
    latex.append("\\centering")
    latex.append("\\caption{Real Cost Measurements}")
    latex.append("\\label{tab:real_costs}")
    latex.append("\\begin{tabular}{lc}")
    latex.append("\\toprule")
    latex.append("\\textbf{Metric} & \\textbf{Value} \\\\")
    latex.append("\\midrule")
    latex.append(f"Total Queries & {total_queries} \\\\")
    latex.append(f"Total Cost & \\${total_cost:.4f} \\\\")
    latex.append(f"Average Cost per Query & \\${total_cost/total_queries:.4f} \\\\")
    latex.append(f"Median Cost per Query & \\${np.median(all_costs):.4f} \\\\")
    latex.append(f"Minimum Cost & \\${min(all_costs):.4f} \\\\")
    latex.append(f"Maximum Cost & \\${max(all_costs):.4f} \\\\")
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    output_path = Path('./paper/tables/real_cost_measurements.tex')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(latex))
    
    print(f"\n✅ Cost table saved to: {output_path}")

if __name__ == "__main__":
    analyze_costs()