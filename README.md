# Cost-Aware Visual Router for Multimodal Document Retrieval

A production-ready implementation of a learned routing system that dynamically selects between visual, text, parametric, and hybrid retrieval paths based on cost-aware utility prediction and uncertainty-aware escalation.

## Problem Statement

Existing Retrieval-Augmented Generation (RAG) systems for multimodal documents indiscriminately apply expensive vision-language retrieval (e.g., ColPali, NeMo Retriever) to all queries, resulting in:
- **Prohibitive computational costs** (visual retrieval consumes 20× more resources than text-based methods)
- **Unnecessary latency** for queries that could be answered through cheaper parametric knowledge or text-only retrieval
- **Static heuristics** that fail to account for query-specific utility of visual information

## Solution

This implementation provides:

1. **Learned Utility Prediction Model** - Estimates expected quality gain per unit cost for each retrieval path
2. **Uncertainty-Aware Escalation** - Combines epistemic, aleatoric, and generation uncertainty for cost-efficient escalation
3. **Dynamic Training Pipeline** - Generates ground-truth routing labels offline without human annotation
4. **Production Router** - Ready for ColPali/NeMo-based document RAG deployment

## Research Objectives Achieved

✅ **Objective 1**: Developed a Learned Utility Prediction Model with multi-task learning for path classification and utility regression with cost-sensitive weighting

✅ **Objective 2**: Designed an Uncertainty-Aware Escalation Mechanism combining epistemic, aleatoric, and generation uncertainty

✅ **Objective 3**: Created a Dynamic Training Data Collection Pipeline that generates ground-truth routing labels by executing all retrieval paths offline

✅ **Objective 4**: Implemented a Production-Ready Visual Router for ColPali/NeMo-based document RAG

✅ **Objective 5**: Established Evaluation Protocols and Benchmarks for cost-aware multimodal RAG routing

## Architecture

```
Query → Query Embedding → Utility Prediction Model → Router Decision
                                      ↓
                              Path Selection
                                      ↓
                    ┌─────────────────┴─────────────────┐
                    ↓                                   ↓
            Cheap Path (e.g., Text)          Expensive Path (e.g., Visual)
                    ↓                                   ↓
            Uncertainty Check ←──────────────────────────┘
                    ↓
            Escalation if needed
                    ↓
            Final Result
```

## Installation

```bash
pip install torch torchvision numpy matplotlib tqdm scikit-learn
```

## Quick Start

### 1. Basic Usage

```python
from cost_aware_visual_router import (
    CostAwareVisualRouter,
    UtilityPredictionModel,
    DEFAULT_PATH_COSTS
)

# Create model
model = UtilityPredictionModel(
    embedding_dim=768,
    hidden_dim=256,
    num_paths=4
)

# Create router
router = CostAwareVisualRouter(
    model=model,
    path_costs=DEFAULT_PATH_COSTS,
    uncertainty_threshold=0.3
)

# Route a query
query = "What is the revenue in Q3?"
query_embedding = get_query_embedding(query)  # Your embedding function

result, decision = router.route(
    query=query,
    query_embedding=query_embedding,
    document_store=your_document_store,
    evaluator=your_quality_evaluator
)

print(f"Selected path: {result.path.value}")
print(f"Quality score: {result.quality_score}")
print(f"Cost: {result.cost}")
```

### 2. Training

```python
from train_router import train_router, RouterDataset
from torch.utils.data import DataLoader

# Load training data
train_dataset = RouterDataset('./training_data')
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Train model
config = {
    'embedding_dim': 768,
    'hidden_dim': 256,
    'num_paths': 4,
    'dropout': 0.1,
    'batch_size': 32,
    'num_epochs': 50,
    'lr': 1e-4,
    'weight_decay': 1e-5,
}

model = train_router(config, train_loader)
```

### 3. Evaluation

```python
from evaluation_metrics import BenchmarkSuite, RoutingEvaluator

# Create evaluator
evaluator = RoutingEvaluator()

# Compute comprehensive metrics
metrics = evaluator.compute_comprehensive_metrics(
    decisions=router_decisions,
    results=router_results,
    ground_truth_paths=ground_truth_paths,
    baseline_results=baseline_results
)

print(f"Routing accuracy: {metrics.routing_accuracy:.2%}")
print(f"Cost reduction vs visual: {metrics.cost_reduction_vs_visual:.1%}")
print(f"Quality vs visual: {metrics.quality_vs_visual:.1%}")
```

## Components

### 1. Utility Prediction Model

Multi-task neural network that predicts:
- **Optimal retrieval path** (classification)
- **Expected utility gain** for each path (regression)
- **Router confidence** (uncertainty estimation)

```python
from cost_aware_visual_router import UtilityPredictionModel

model = UtilityPredictionModel(
    embedding_dim=768,
    hidden_dim=256,
    num_paths=4,
    dropout=0.1
)

outputs = model(query_embedding, path_costs)
# Returns: path_logits, path_probs, utilities, confidence
```

### 2. Uncertainty-Aware Escalation

Combines three types of uncertainty:
- **Epistemic uncertainty**: Router model confidence (Monte Carlo dropout)
- **Aleatoric uncertainty**: Retrieval score variance
- **Generation uncertainty**: Token perplexity

```python
from cost_aware_visual_router import UncertaintyAwareEscalation

escalation = UncertaintyAwareEscalation(
    uncertainty_threshold=0.3
)

should_escalate, reason = escalation.should_escalate(
    current_path=current_path,
    router_confidence=0.2,
    retrieval_scores=[0.8, 0.3, 0.1],
    generation_uncertainty=0.4
)
```

### 3. Dynamic Training Pipeline

Generates training data by executing all retrieval paths offline:

```python
from cost_aware_visual_router import DynamicTrainingPipeline

pipeline = DynamicTrainingPipeline(
    path_costs=DEFAULT_PATH_COSTS,
    quality_weight=0.7,
    cost_weight=0.3
)

sample = pipeline.generate_training_sample(
    query="What is the revenue?",
    query_embedding=query_embedding,
    document_store=document_store,
    evaluator=quality_evaluator
)
```

### 4. Production Router

Main routing system with statistics tracking:

```python
from cost_aware_visual_router import CostAwareVisualRouter

router = CostAwareVisualRouter(
    model=model,
    path_costs=DEFAULT_PATH_COSTS,
    uncertainty_threshold=0.3
)

# Route queries
for query in queries:
    result, decision = router.route(
        query=query,
        query_embedding=get_embedding(query),
        document_store=doc_store,
        evaluator=evaluator
    )

# Get statistics
stats = router.get_statistics()
print(f"Average cost: {stats['average_cost']:.3f}")
print(f"Average quality: {stats['average_quality']:.3f}")
print(f"Cost efficiency: {stats['cost_efficiency']:.3f}")
```

## Retrieval Paths

| Path | Description | Cost (normalized) | Latency (ms) |
|------|-------------|-------------------|--------------|
| Parametric | LLM internal knowledge only | 0.01 | 50 |
| Text Only | Text-only retrieval | 0.05 | 200 |
| Visual Only | Visual-only retrieval | 1.0 | 2000 |
| Hybrid | Combined text + visual | 1.2 | 2500 |

## Evaluation Metrics

### Routing Metrics
- **Routing Accuracy**: Percentage of correct path selections
- **Path Accuracy**: Accuracy per path type
- **Path Distribution**: Distribution of selected paths

### Cost Metrics
- **Average Cost**: Mean cost across all queries
- **Cost Reduction vs Visual**: Percentage cost reduction compared to always using visual
- **Cost Efficiency**: Quality achieved per unit cost

### Quality Metrics
- **Average Quality**: Mean quality score
- **Quality vs Visual**: Quality relative to always using visual

### Escalation Metrics
- **Escalation Rate**: Percentage of queries that triggered escalation
- **Escalation Success Rate**: Percentage of escalations that improved quality
- **Average Escalations**: Mean number of escalations per query

## Benchmarks

The system is evaluated against the following baselines:

1. **Always Parametric** - Always use LLM internal knowledge
2. **Always Text** - Always use text-only retrieval
3. **Always Visual** - Always use visual-only retrieval
4. **Always Hybrid** - Always use hybrid retrieval
5. **Random Routing** - Random path selection
6. **Static Heuristic** - Keyword-based routing

## Target Performance

Based on the research objectives:

- **Accuracy**: ≥85% of always-using-visual-retrieval
- **Cost Reduction**: ≥40% average cost reduction
- **Escalation**: ≤5% accuracy degradation on complex visual queries

## File Structure

```
.
├── cost_aware_visual_router.py    # Main implementation
├── train_router.py                # Training script
├── evaluation_metrics.py          # Evaluation and benchmarks
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

## Configuration

### Path Costs

Customize path costs based on your system:

```python
from cost_aware_visual_router import PathCost, RetrievalPath

custom_costs = {
    RetrievalPath.PARAMETRIC: PathCost(
        path=RetrievalPath.PARAMETRIC,
        compute_cost=0.01,
        latency_ms=50,
        token_cost=0.0
    ),
    # ... configure other paths
}
```

### Model Hyperparameters

```python
config = {
    'embedding_dim': 768,      # Query embedding dimension
    'hidden_dim': 256,          # Hidden layer dimension
    'num_paths': 4,            # Number of retrieval paths
    'dropout': 0.1,            # Dropout rate
    'uncertainty_threshold': 0.3,  # Escalation threshold
}
```

## Integration with ColPali/NeMo

To integrate with ColPali or NeMo Retriever:

```python
class ColPaliDocumentStore(DocumentStore):
    """Document store using ColPali for visual retrieval"""

    def retrieve_visual(self, query: str):
        # Use ColPali for visual retrieval
        results = self.colpali_client.retrieve(query, k=5)
        documents = [r['document'] for r in results]
        scores = [r['score'] for r in results]
        return documents, scores

    def retrieve_hybrid(self, query: str):
        # Combine text and visual retrieval
        text_docs, text_scores = self.retrieve_text(query)
        visual_docs, visual_scores = self.retrieve_visual(query)

        # Merge and re-rank
        merged = self.merge_results(text_docs, text_scores,
                                   visual_docs, visual_scores)
        return merged['documents'], merged['scores']
```

## Advanced Features

### Cost-Sensitive Loss Weighting

The model uses cost-sensitive weighting to optimize for cost-quality trade-offs:

```python
loss_weights = {
    'classification': 1.0,
    'utility': 0.5,
    'confidence': 0.3
}
```

### Multi-Task Learning

The model is trained on three tasks simultaneously:
1. Path classification (cross-entropy loss)
2. Utility regression (MSE loss)
3. Confidence estimation (KL divergence loss)

### Uncertainty Quantification

Three types of uncertainty are estimated:
- **Epistemic**: Model uncertainty (Monte Carlo dropout)
- **Aleatoric**: Data uncertainty (score variance)
- **Generation**: Output uncertainty (perplexity)

## Citation

If you use this implementation, please cite:

```bibtex
@article{cost_aware_visual_router,
  title={Cost-Aware Visual Router for Multimodal Document Retrieval},
  author={Your Name},
  year={2024}
}
```

## License

This implementation is provided for research and educational purposes.

## Contributing

Contributions are welcome! Areas for improvement:
- Additional retrieval path types
- More sophisticated uncertainty estimation
- Advanced escalation strategies
- Real-world dataset integration
- Production deployment optimizations

## Contact

For questions or issues, please open an issue on the repository.