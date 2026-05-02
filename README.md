# Cost-Aware Visual Router (CAVR)
## Dynamic Query Routing for Cost-Efficient Multimodal RAG

## Overview

CAVR is a learned routing framework that dynamically selects among four retrieval paths—parametric (LLM knowledge), text-only, visual-only (ColPali), and hybrid—for cost-efficient multimodal document retrieval.

### Key Results
- **86.2%** overall routing accuracy on balanced DocVQA test set
- **88.2%** visual query accuracy | **94.2%** hybrid query accuracy
- **$0.00019** average cost per query (97.5% below estimates)

## Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/cost-aware-visual-router.git
cd cost-aware-visual-router

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt






Quick Start
python
from colpali_router_demo import CompleteVisualRouter

# Initialize router
router = CompleteVisualRouter()

# Route a query
query = "What does a neural network diagram look like?"
result = router.route_and_retrieve(query)
print(f"Path: {result['path_name']}")
print(f"Confidence: {result['confidence']:.3f}")


-----------------------------------------------

Training a New Router

# 1. Prepare balanced dataset
python scripts/create_balanced_dataset_v2.py

# 2. Train the router
python scripts/train_balanced_router_v2.py

# 3. Evaluate
python scripts/evaluate_balanced_router.py

# 4. Generate paper figures
python scripts/generate_paper_tables.py

---------------------------------------

Datasets

We use the following benchmarks:

DocVQA - Document Visual Question Answering

InfoVQA - Infographics VQA


Results
Metric	Value
Overall Accuracy	86.2%
Visual Accuracy	88.2%
Hybrid Accuracy	94.2%
Avg Cost/Query	$0.00019



@article{rashid2026cost,
  title={Cost-Aware Visual Router (CAVR): Dynamic Query Routing for Cost-Efficient Multimodal RAG},
  author={Rashid, Muhammad Ahsan and Shabbir, Sohail},
  journal={arXiv preprint},
  year={2026}
}

