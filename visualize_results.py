import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def visualize_test_results(csv_file='router_test_results_20260426_001256.csv'):
    """Create visualizations of test results from CSV"""
    
    # Load CSV
    df = pd.read_csv(csv_file)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Confidence histogram
    axes[0, 0].hist(df['confidence'], bins=20, edgecolor='black')
    axes[0, 0].set_title('Confidence Distribution')
    axes[0, 0].set_xlabel('Confidence')
    axes[0, 0].set_ylabel('Frequency')
    
    # 2. Path distribution
    axes[0, 1].hist(df['predicted_path'], bins=4, edgecolor='black')
    axes[0, 1].set_title('Path Distribution')
    axes[0, 1].set_xlabel('Path ID')
    axes[0, 1].set_ylabel('Count')
    
    # 3. Confidence by path (boxplot)
    path_data = []
    for path in range(4):
        path_conf = df[df['predicted_path'] == path]['confidence']
        if not path_conf.empty:
            path_data.append(path_conf)
        else:
            path_data.append([0])  # fallback
    
    axes[1, 0].boxplot(path_data)
    axes[1, 0].set_title('Confidence by Path')
    axes[1, 0].set_xlabel('Path ID')
    axes[1, 0].set_ylabel('Confidence')
    
    # 4. Domain performance
    domain_avg = df.groupby('category')['confidence'].mean()
    
    axes[1, 1].bar(domain_avg.index, domain_avg.values)
    axes[1, 1].set_title('Performance by Domain')
    axes[1, 1].set_xlabel('Domain')
    axes[1, 1].set_ylabel('Average Confidence')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('router_test_visualization.png', dpi=150)
    plt.show()
    
    print("Visualization saved to router_test_visualization.png")


if __name__ == "__main__":
    visualize_test_results()