# """
# Generate training data with clear, strong preferences for each path
# FIXED VERSION
# """

# import json
# import numpy as np
# from pathlib import Path
# from sentence_transformers import SentenceTransformer
# from tqdm import tqdm
# import random

# def generate_strong_training_data(num_samples=2000):
#     """Generate training data with clear path preferences"""
    
#     output_dir = Path('./strong_training_data')
#     output_dir.mkdir(exist_ok=True)
    
#     print("Loading encoder...")
#     encoder = SentenceTransformer('all-mpnet-base-v2')
    
#     # Define queries with CLEAR preferences for each path type
#     path_queries = {
#         0: {  # Parametric - Simple factual questions (should use LLM knowledge)
#             'queries': [
#                 "What is the capital of France?",
#                 "Who wrote Romeo and Juliet?",
#                 "What is 2+2?",
#                 "When was Python created?",
#                 "What is the chemical symbol for gold?",
#                 "Who is the CEO of Microsoft?",
#                 "What year did World War II end?",
#                 "How many continents are there?",
#                 "What is the largest ocean?",
#                 "Who painted the Mona Lisa?",
#             ],
#             'utility_boost': 0.95,
#             'cost': 0.0001
#         },
#         1: {  # Text-Only - Technical/Programming questions (needs document search)
#             'queries': [
#                 "How do I implement quicksort in Python?",
#                 "What are the best practices for REST API design?",
#                 "Explain the difference between SQL and NoSQL databases",
#                 "How to optimize React rendering performance?",
#                 "What is dependency injection in Spring Boot?",
#                 "How to use async/await in JavaScript?",
#                 "Explain garbage collection in Java",
#                 "What is the difference between Git merge and rebase?",
#                 "How to write unit tests in pytest?",
#                 "Explain Docker container networking",
#             ],
#             'utility_boost': 0.95,
#             'cost': 0.0005
#         },
#         2: {  # Visual-Only - Image/diagram questions (needs visual retrieval)
#             'queries': [
#                 "What does the Eiffel Tower look like?",
#                 "Show me a diagram of neural network architecture",
#                 "What does the Mona Lisa painting look like?",
#                 "Compare images of cats and dogs",
#                 "What is the shape of a DNA double helix?",
#                 "Show me examples of Art Deco architecture",
#                 "What does a human heart look like?",
#                 "Show me different types of flowers",
#                 "What does the solar system look like?",
#                 "Show me images of famous landmarks",
#             ],
#             'utility_boost': 0.95,
#             'cost': 0.01
#         },
#         3: {  # Hybrid - Complex questions needing both text and images
#             'queries': [
#                 "Explain how neural networks work with diagrams",
#                 "Show me step-by-step how to bake a cake with instructions",
#                 "Explain the water cycle with visual examples",
#                 "Show me how to perform CPR with illustrations",
#                 "Explain how solar panels work with diagrams",
#                 "Show me how to change a tire with pictures",
#                 "Explain the process of photosynthesis with images",
#                 "Show me how to solve a Rubik's cube with visual steps",
#                 "Explain how a car engine works with diagrams",
#                 "Show me how to do yoga poses with images",
#             ],
#             'utility_boost': 0.95,
#             'cost': 0.011
#         }
#     }
    
#     samples_per_path = num_samples // 4
#     sample_id = 0
    
#     print(f"\nGenerating {num_samples} training samples with strong preferences...")
    
#     for path_id, path_info in path_queries.items():
#         print(f"\nGenerating for Path {path_id} ({len(path_info['queries'])} query templates)...")
        
#         for i in range(samples_per_path):
#             # Get base query
#             base_query = path_info['queries'][i % len(path_info['queries'])]
            
#             # Create variations
#             variations = [
#                 base_query,
#                 f"Explain: {base_query}",
#                 f"Tell me about {base_query}",
#                 f"What is {base_query}?",
#                 f"Help me understand {base_query}",
#             ]
#             query = variations[i % len(variations)]
            
#             # Add uniqueness
#             if i % 3 == 0:
#                 query = query + " in detail"
            
#             # Get embedding
#             embedding = encoder.encode(query).astype(np.float32)
            
#             # Create STRONG utility preferences
#             utilities = np.random.uniform(0.05, 0.15, 4)  # Low baseline
#             utilities[path_id] = path_info['utility_boost']  # Very high for optimal path
            
#             # Add small noise
#             utilities = utilities + np.random.normal(0, 0.03, 4)
#             utilities = np.clip(utilities, 0, 1)
            
#             # Costs
#             costs = [0.0001, 0.0005, 0.01, 0.011]
            
#             # Quality scores (parallel to utilities)
#             quality = utilities.copy()
            
#             sample = {
#                 'query': query,
#                 'query_embedding': embedding.tolist(),
#                 'optimal_path': path_id,
#                 'path_utilities': utilities.tolist(),
#                 'path_costs': costs,
#                 'quality_scores': quality.tolist(),
#             }
            
#             # Save
#             output_path = output_dir / f"sample_{sample_id:06d}.json"
#             with open(output_path, 'w') as f:
#                 json.dump(sample, f, indent=2)
            
#             sample_id += 1
            
#             if (i + 1) % 200 == 0:
#                 print(f"  Generated {i+1}/{samples_per_path} for path {path_id}")
    
#     print(f"\n✅ Generated {sample_id} training samples in {output_dir}")
    
#     # FIXED: Verify distribution - properly open files
#     print("\n📊 Verifying generated data...")
#     samples = []
#     for file_path in output_dir.glob("*.json"):
#         with open(file_path, 'r') as f:  # FIXED: open file properly
#             samples.append(json.load(f))
    
#     if samples:
#         # Check path distribution
#         path_counts = {}
#         for s in samples[:200]:  # Check first 200
#             path = s['optimal_path']
#             path_counts[path] = path_counts.get(path, 0) + 1
        
#         print("  Sample distribution (first 200 files):")
#         for path, count in sorted(path_counts.items()):
#             print(f"    Path {path}: {count} samples")
        
#         # Check utility distinction
#         print("\n  Utility verification:")
#         for path in range(4):
#             sample_utils = [s['path_utilities'][path] for s in samples[:50]]
#             print(f"    Path {path} avg utility: {np.mean(sample_utils):.3f}")
#     else:
#         print("  No samples found to verify")
    
#     return output_dir

# if __name__ == '__main__':
#     generate_strong_training_data(2000)







"""
Generate training data with clear, strong preferences for each path
Reads training queries from external text file
"""

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import random
import argparse

def load_training_queries_from_file(filepath="training_queries.txt"):
    """
    Load training queries from text file.
    Format: PathID|Category|Query
    Lines starting with # are ignored
    """
    queries_by_path = {
        0: {'queries': [], 'utility_boost': 0.95, 'cost': 0.0001, 'name': 'Parametric'},
        1: {'queries': [], 'utility_boost': 0.95, 'cost': 0.0005, 'name': 'Text-Only'},
        2: {'queries': [], 'utility_boost': 0.95, 'cost': 0.01, 'name': 'Visual-Only'},
        3: {'queries': [], 'utility_boost': 0.95, 'cost': 0.011, 'name': 'Hybrid'}
    }
    
    file_path = Path(filepath)
    
    if not file_path.exists():
        print(f"❌ File not found: {filepath}")
        print(f"   Creating sample file: {filepath}")
        create_sample_training_queries_file(filepath)
        return queries_by_path
    
    print(f"📖 Loading training queries from {filepath}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Parse path_id, category, query
            if '|' in line:
                parts = line.split('|', 2)
                if len(parts) >= 3:
                    try:
                        path_id = int(parts[0].strip())
                        category = parts[1].strip()
                        query = parts[2].strip()
                        
                        if path_id in queries_by_path:
                            queries_by_path[path_id]['queries'].append(query)
                        else:
                            print(f"⚠️ Warning: Line {line_num} has invalid path_id: {path_id}")
                    except ValueError:
                        print(f"⚠️ Warning: Line {line_num} has invalid path_id format: {parts[0]}")
                else:
                    print(f"⚠️ Warning: Line {line_num} has invalid format (expected PathID|Category|Query)")
            else:
                print(f"⚠️ Warning: Line {line_num} has invalid format (missing '|'): {line[:50]}")
    
    # Print summary
    print("\n📊 Loaded training queries summary:")
    for path_id, info in queries_by_path.items():
        print(f"  Path {path_id} ({info['name']}): {len(info['queries'])} queries")
    
    # Validate that each path has at least one query
    for path_id, info in queries_by_path.items():
        if len(info['queries']) == 0:
            print(f"⚠️ Warning: Path {path_id} ({info['name']}) has no queries!")
            # Add default query
            default_queries = {
                0: ["What is the capital of France?"],
                1: ["How to implement binary search in Python?"],
                2: ["What does the Eiffel Tower look like?"],
                3: ["Explain how neural networks work with diagrams"]
            }
            info['queries'] = default_queries.get(path_id, ["Sample query"])
            print(f"   Added default query: {info['queries'][0]}")
    
    return queries_by_path

def create_sample_training_queries_file(filepath="training_queries.txt"):
    """Create a sample training queries file if none exists"""
    sample_content = """# Training queries for Cost-Aware Visual Router
# Format: PathID|Category|Query
# PathID: 0=Parametric, 1=Text-Only, 2=Visual-Only, 3=Hybrid
# Lines starting with # are ignored

# Parametric queries (Path 0) - Simple factual questions
0|Parametric|What is the capital of France?
0|Parametric|Who wrote Romeo and Juliet?
0|Parametric|What is 2+2?
0|Parametric|When was Python created?
0|Parametric|What is the chemical symbol for gold?

# Text-Only queries (Path 1) - Technical/Programming
1|Text-Only|How do I implement quicksort in Python?
1|Text-Only|What are the best practices for REST API design?
1|Text-Only|Explain the difference between SQL and NoSQL databases
1|Text-Only|How to optimize React rendering performance?
1|Text-Only|What is dependency injection in Spring Boot?

# Visual-Only queries (Path 2) - Image/diagram questions
2|Visual-Only|What does the Eiffel Tower look like?
2|Visual-Only|Show me a diagram of neural network architecture
2|Visual-Only|What does the Mona Lisa painting look like?
2|Visual-Only|Compare images of cats and dogs
2|Visual-Only|What is the shape of a DNA double helix?

# Hybrid queries (Path 3) - Need both text and visuals
3|Hybrid|Explain how neural networks work with diagrams
3|Hybrid|Show me step-by-step how to bake a cake with instructions
3|Hybrid|Explain the water cycle with visual examples
3|Hybrid|Show me how to perform CPR with illustrations
3|Hybrid|Explain how solar panels work with diagrams
"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(sample_content)
    print(f"✅ Created sample file: {filepath}")
    print("   Edit this file to add your own training queries")

def generate_strong_training_data(num_samples=4000, queries_file="training_queries.txt"):
    """Generate training data with clear path preferences using external query file"""
    
    output_dir = Path('./strong_training_data')
    output_dir.mkdir(exist_ok=True)
    
    print("Loading encoder...")
    encoder = SentenceTransformer('all-mpnet-base-v2')
    
    # Load queries from file
    path_queries = load_training_queries_from_file(queries_file)
    
    samples_per_path = num_samples // 4
    sample_id = 0
    
    print(f"\n🎯 Generating {num_samples} training samples with strong preferences...")
    print(f"   {samples_per_path} samples per path")
    
    # Variations for data augmentation
    variation_templates = [
        "{query}",
        "Explain: {query}",
        "Tell me about {query}",
        "What is {query}?",
        "Help me understand {query}",
        "Can you explain {query}?",
        "I need information about {query}",
        "Please explain {query} in detail"
    ]
    
    for path_id, path_info in path_queries.items():
        queries_list = path_info['queries']
        if not queries_list:
            print(f"⚠️ Skipping Path {path_id} - no queries available")
            continue
            
        print(f"\n📂 Generating for Path {path_id} ({path_info['name']}) - {len(queries_list)} query templates...")
        
        for i in range(samples_per_path):
            # Get base query (cycle through available queries)
            base_query = queries_list[i % len(queries_list)]
            
            # Create variations for data augmentation
            variation_template = variation_templates[i % len(variation_templates)]
            query = variation_template.format(query=base_query)
            
            # Add uniqueness with random modifier
            modifiers = ["", " in detail", " with examples", " step by step", " clearly"]
            if i % 5 == 0:
                query = query + modifiers[i // 5 % len(modifiers)]
            
            # Get embedding
            embedding = encoder.encode(query).astype(np.float32)
            
            # Create STRONG utility preferences
            utilities = np.random.uniform(0.05, 0.15, 4)  # Low baseline
            utilities[path_id] = path_info['utility_boost']  # Very high for optimal path
            
            # Add small noise
            utilities = utilities + np.random.normal(0, 0.03, 4)
            utilities = np.clip(utilities, 0, 1)
            
            # Costs
            costs = [0.0001, 0.0005, 0.01, 0.011]
            
            # Quality scores (parallel to utilities)
            quality = utilities.copy()
            
            sample = {
                'query': query,
                'query_embedding': embedding.tolist(),
                'optimal_path': path_id,
                'path_utilities': utilities.tolist(),
                'path_costs': costs,
                'quality_scores': quality.tolist(),
            }
            
            # Save
            output_path = output_dir / f"sample_{sample_id:06d}.json"
            with open(output_path, 'w') as f:
                json.dump(sample, f, indent=2)
            
            sample_id += 1
            
            if (i + 1) % max(1, samples_per_path // 10) == 0:
                progress = (i + 1) / samples_per_path * 100
                print(f"  Progress: {i+1}/{samples_per_path} ({progress:.0f}%)")
    
    print(f"\n✅ Generated {sample_id} training samples in {output_dir}")
    
    # Verify distribution
    verify_generated_data(output_dir)
    
    return output_dir

def verify_generated_data(output_dir):
    """Verify the generated training data"""
    print("\n📊 Verifying generated data...")
    samples = []
    for file_path in output_dir.glob("*.json"):
        with open(file_path, 'r') as f:
            samples.append(json.load(f))
    
    if samples:
        # Check path distribution
        path_counts = {}
        for s in samples[:200]:  # Check first 200
            path = s['optimal_path']
            path_counts[path] = path_counts.get(path, 0) + 1
        
        print("  Sample distribution (first 200 files):")
        for path, count in sorted(path_counts.items()):
            path_names = {0: "Parametric", 1: "Text-Only", 2: "Visual-Only", 3: "Hybrid"}
            print(f"    Path {path} ({path_names.get(path, 'Unknown')}): {count} samples ({count/2:.1f}%)")
        
        # Check utility distinction
        print("\n  Utility verification (optimal path should be highest):")
        for path in range(4):
            sample_utils = [s['path_utilities'][path] for s in samples[:50]]
            avg_util = np.mean(sample_utils)
            is_optimal = path == list(path_counts.keys())[0] if path_counts else False
            marker = "⭐" if is_optimal else "  "
            print(f"    {marker} Path {path}: avg utility = {avg_util:.3f}")
        
        # Check for any issues
        issues = 0
        for s in samples[:100]:
            optimal = s['optimal_path']
            utilities = s['path_utilities']
            if utilities[optimal] < max(utilities):
                issues += 1
        if issues > 0:
            print(f"\n  ⚠️ Warning: {issues}/100 samples have suboptimal utility assignment")
        else:
            print("\n  ✅ All samples have correct utility assignment")
    else:
        print("  No samples found to verify")

def generate_custom_dataset(num_samples=4000, queries_file="training_queries.txt"):
    """Wrapper function to generate dataset with custom parameters"""
    return generate_strong_training_data(num_samples, queries_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate training data for visual router')
    parser.add_argument('--num_samples', type=int, default=4000,
                       help='Number of training samples to generate (default: 4000)')
    parser.add_argument('--queries_file', type=str, default='training_queries.txt',
                       help='Path to training queries file (default: training_queries.txt)')
    parser.add_argument('--output_dir', type=str, default='./strong_training_data',
                       help='Output directory for training data (default: ./strong_training_data)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("TRAINING DATA GENERATOR")
    print("="*60)
    print(f"  Samples: {args.num_samples}")
    print(f"  Queries file: {args.queries_file}")
    print(f"  Output dir: {args.output_dir}")
    print("="*60 + "\n")
    
    generate_strong_training_data(args.num_samples, args.queries_file)
    
    
    