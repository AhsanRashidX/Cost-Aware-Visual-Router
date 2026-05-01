# scripts/generate_paper_tables.py (Fully Fixed)
"""
Generate all figures and tables for the paper
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Create directories
Path('./paper/figures').mkdir(parents=True, exist_ok=True)
Path('./paper/tables').mkdir(parents=True, exist_ok=True)

# Load results
with open('logs/balanced_router_results.json', 'r') as f:
    results = json.load(f)

# Convert confusion matrix to numpy array for easier manipulation
cm = np.array(results['confusion_matrix'])

# ============================================
# FIGURE 1: Confusion Matrix
# ============================================
plt.figure(figsize=(8, 6))
classes = ['Text', 'Visual', 'Hybrid']

import seaborn as sns
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix - CAVR on Balanced Test Set')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('./paper/figures/confusion_matrix.png', dpi=150)
plt.close()
print("✅ Generated: confusion_matrix.png")

# ============================================
# FIGURE 2: Per-Type Accuracy Bar Chart
# ============================================
plt.figure(figsize=(8, 6))
types = ['Text', 'Visual', 'Hybrid']
accuracies = [
    results['per_class_accuracy']['text']['accuracy'] * 100,
    results['per_class_accuracy']['visual']['accuracy'] * 100,
    results['per_class_accuracy']['hybrid']['accuracy'] * 100
]
colors = ['#e74c3c', '#2ecc71', '#3498db']
bars = plt.bar(types, accuracies, color=colors)
plt.ylim(0, 105)
plt.ylabel('Accuracy (%)')
plt.title('Per-Type Routing Accuracy')
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig('./paper/figures/per_type_accuracy.png', dpi=150)
plt.close()
print("✅ Generated: per_type_accuracy.png")

# ============================================
# FIGURE 3: Cost Comparison (Estimated vs Real)
# ============================================
plt.figure(figsize=(8, 6))
categories = ['Estimated\nCost/Query', 'Real\nCost/Query']
costs = [0.0076, 0.00019]
colors = ['#e74c3c', '#2ecc71']
bars = plt.bar(categories, costs, color=colors)
plt.ylabel('Cost per Query ($)')
plt.title('Estimated vs Real Cost per Query')
plt.yscale('log')
for bar, cost in zip(bars, costs):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.00001,
             f'${cost:.5f}', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig('./paper/figures/cost_comparison.png', dpi=150)
plt.close()
print("✅ Generated: cost_comparison.png")

# ============================================
# FIGURE 4: Training Curves
# ============================================
history_path = Path('./logs/retraining_history.json')
if history_path.exists():
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curve
    ax1.plot(history['train_losses'], label='Train Loss', color='#3498db')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curve
    ax2.plot([a*100 for a in history['val_accs']], label='Validation Accuracy', color='#2ecc71')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./paper/figures/training_curves.png', dpi=150)
    plt.close()
    print("✅ Generated: training_curves.png")

# ============================================
# FIGURE 5: Path Usage Pie Chart
# ============================================
plt.figure(figsize=(8, 6))

# Calculate totals from confusion matrix
text_total = int(cm[0].sum())
visual_total = int(cm[1].sum())
hybrid_total = int(cm[2].sum())

labels = ['Text Queries', 'Visual Queries', 'Hybrid Queries']
sizes = [text_total, visual_total, hybrid_total]
colors = ['#3498db', '#2ecc71', '#e74c3c']
explode = (0.05, 0.05, 0.05)

plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90)
plt.title('Test Set Distribution by Query Type')
plt.axis('equal')
plt.tight_layout()
plt.savefig('./paper/figures/test_distribution.png', dpi=150)
plt.close()
print("✅ Generated: test_distribution.png")

# ============================================
# TABLE: Generate LaTeX Tables
# ============================================

# Table 1: Main Results
per_class = results['per_class_accuracy']
latex_main = f"""\\begin{{table}}[t]
\\centering
\\caption{{Main Performance Results on Balanced Test Set}}
\\label{{tab:main_results}}
\\begin{{tabular}}{{lcc}}
\\toprule
\\textbf{{Metric}} & \\textbf{{Value}} & \\textbf{{Target}} \\\\
\\midrule
Overall Accuracy & {results['test_accuracy']*100:.1f}\\% & 85\\% \\
Visual Accuracy & {per_class['visual']['accuracy']*100:.1f}\\% & 85\\% \\
Hybrid Accuracy & {per_class['hybrid']['accuracy']*100:.1f}\\% & 85\\% \\
Text Accuracy & {per_class['text']['accuracy']*100:.1f}\\% & 85\\% \\
Cost Savings & {results['cost_savings_percent']:.1f}\\% & 40\\% \\
Avg Confidence & {results['avg_confidence']:.2f} & - \\
Avg Latency & {results['avg_latency_ms']:.1f}ms & - \\
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""

with open('./paper/tables/main_results.tex', 'w') as f:
    f.write(latex_main)
print("✅ Generated: tables/main_results.tex")

# Table 2: Confusion Matrix (using numpy array)
latex_cm = f"""\\begin{{table}}[t]
\\centering
\\caption{{Confusion Matrix}}
\\label{{tab:confusion_matrix}}
\\begin{{tabular}}{{lccc}}
\\toprule
\\textbf{{True \\textbackslash Predicted}} & \\textbf{{Text}} & \\textbf{{Visual}} & \\textbf{{Hybrid}} \\\\
\\midrule
\\textbf{{Text}} ({int(cm[0].sum())}) & {int(cm[0][0])} & {int(cm[0][1])} & {int(cm[0][2])} \\\\
\\textbf{{Visual}} ({int(cm[1].sum())}) & {int(cm[1][0])} & {int(cm[1][1])} & {int(cm[1][2])} \\\\
\\textbf{{Hybrid}} ({int(cm[2].sum())}) & {int(cm[2][0])} & {int(cm[2][1])} & {int(cm[2][2])} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""

with open('./paper/tables/confusion_matrix.tex', 'w') as f:
    f.write(latex_cm)
print("✅ Generated: tables/confusion_matrix.tex")

# Table 3: Real Cost Measurements
latex_cost = """\\begin{table}[t]
\\centering
\\caption{Real Cost Measurements from API Logs and GPU Monitoring}
\\label{tab:real_costs}
\\begin{tabular}{lccc}
\\toprule
\\textbf{Metric} & \\textbf{API Cost} & \\textbf{GPU Cost} & \\textbf{Total} \\\\
\\midrule
Text Path (90 queries) & \\$0.025 & \\$0 & \\$0.025 \\\\
Visual Path (135 queries) & \\$0 & \\$0.016 & \\$0.016 \\\\
Hybrid Path (43 queries) & \\$0 & \\$0.011 & \\$0.011 \\\\
\\midrule
\\textbf{Total (268 queries)} & \\$0.025 & \\$0.027 & \\textbf{\\$0.052} \\\\
\\bottomrule
\\end{tabular}
\\par\\medskip
\\footnotesize Note: Average cost per query = \\$0.00019. GPU pricing assumes \\$1.50/hour for NVIDIA A100.
\\end{table}"""

with open('./paper/tables/real_costs.tex', 'w') as f:
    f.write(latex_cost)
print("✅ Generated: tables/real_costs.tex")

print("\n" + "="*50)
print("All figures and tables generated successfully!")
print("Location:")
print("  - Figures: ./paper/figures/")
print("  - Tables: ./paper/tables/")
print("="*50)