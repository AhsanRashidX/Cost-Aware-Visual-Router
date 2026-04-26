# ============================================
# NEW: ColPali Retriever for Path 2 (Visual-Only)
# ============================================
import torch
from byaldi import RAGMultiModalModel
from pdf2image import convert_from_path
from pathlib import Path
from typing import List, Dict, Union
import base64
from io import BytesIO

class ColPaliRetriever:
    """
    A document retriever that uses ColPali for visual understanding.
    Treats documents as images, indexing them for efficient visual search.
    """
    def __init__(self, model_name: str = "vidore/colpali-v1.3-hf", index_root: str = ".byaldi"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🚀 Initializing ColPali Retriever on {self.device}...")
        
        # Initialize the ColPali model using the convenient byaldi wrapper
        self.model = RAGMultiModalModel.from_pretrained(model_name, index_root=index_root)
        self.cost = RETRIEVAL_COSTS['visual_only']  # Use your existing cost structure
        self.current_index_name = None

    def index_document(self, doc_path: Union[str, Path], index_name: str):
        """
        Indexes a PDF or image document.
        Args:
            doc_path: Path to a PDF file or directory containing images.
            index_name: A unique name for this index.
        """
        print(f"📑 Indexing document '{doc_path}' as '{index_name}'...")
        self.model.index(
            input_path=str(doc_path),
            index_name=index_name,
            overwrite=True,
            store_collection_with_index=True,  # Stores base64 for later use
        )
        self.current_index_name = index_name
        print(f"✅ Document indexed successfully.")

    def load_index(self, index_name: str):
        """Loads a previously created index."""
        print(f"📂 Loading existing index '{index_name}'...")
        self.model = RAGMultiModalModel.from_index(index_name)
        self.current_index_name = index_name
        print(f"✅ Index '{index_name}' loaded.")

    def retrieve(self, query: str, k: int = 5) -> Dict:
        """
        Retrieves the most relevant document pages for a query.
        Args:
            query: The user's query string.
            k: The number of top results to return.
        Returns:
            A dictionary containing the answer, metadata, and costs.
        """
        start_time = time.time()
        
        if self.current_index_name is None:
            raise ValueError("No index loaded. Call index_document() or load_index() first.")

        # Perform the search using the ColPali model
        results = self.model.search(query, k=k)

        latency = (time.time() - start_time) * 1000  # ms
        
        # Format the results
        retrieved_docs = []
        similarities = []
        for result in results:
            retrieved_docs.append({
                'doc_id': result.doc_id,
                'page_num': result.page_num,
                'score': result.score,
                'metadata': result.metadata,
                'base64': result.base64 if hasattr(result, 'base64') else None
            })
            similarities.append(result.score)

        # Generate a simple answer string from the results
        if results:
            answer = f"Retrieved {len(results)} relevant page(s). Best score: {results[0].score:.3f}"
        else:
            answer = "No relevant documents found."

        return {
            'answer': answer,
            'latency_ms': latency,
            'cost': self.cost.total_cost,
            'retrieved_docs': retrieved_docs,
            'similarities': similarities
        }