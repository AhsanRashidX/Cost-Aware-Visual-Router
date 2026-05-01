"""
Real DocVQA Dataset Loader (Not Synthetic)
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
import logging
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DocVQASample:
    """Single DocVQA sample"""
    question_id: int
    question: str
    image_id: str
    image_path: str
    answers: List[str]
    question_type: str  # from dataset if available
    predicted_type: str  # text/visual/hybrid based on keywords
    doc_id: str


class RealDocVQADataset:
    """
    Loader for actual DocVQA dataset (Task 1 - Single Page)
    """
    
    def __init__(self, data_root: str, split: str = 'train', 
                 max_samples: int = None, classify_questions: bool = True):
        """
        Args:
            data_root: Path to data directory (contains annotations/ and images/)
            split: 'train', 'val', or 'test'
            max_samples: Limit number of samples for quick testing
            classify_questions: Auto-classify questions into text/visual/hybrid
        """
        self.data_root = Path(data_root)
        self.split = split
        self.classify_questions = classify_questions
        
        # Load annotations
        self.annotations = self._load_annotations()
        
        # Create samples
        self.samples = self._create_samples()
        
        if max_samples:
            self.samples = self.samples[:max_samples]
        
        # Print statistics
        self._print_stats()
    
    def _load_annotations(self) -> Dict:
        """Load DocVQA annotation JSON file"""
        ann_file = self.data_root / 'annotations' / f'{self.split}.json'
        
        if not ann_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {ann_file}")
        
        with open(ann_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded annotations from {ann_file}")
        return data
    
    def _classify_question(self, question: str, question_type: str = None) -> str:
        """
        Classify question into text, visual, or hybrid
        Based on paper's classification criteria
        """
        if question_type and not self.classify_questions:
            # Use dataset's question type if available
            if 'text' in question_type.lower():
                return 'text'
            elif 'visual' in question_type.lower():
                return 'visual'
        
        q_lower = question.lower()
        
        # Text-only keywords (factual, extraction, counting)
        text_keywords = [
            'what is', 'who is', 'when was', 'where is', 'how many',
            'what year', 'what date', 'what is the name', 'what is the title',
            'what does the text', 'what word', 'what phrase', 'what number',
            'how much', 'how long', 'what is the value', 'extract', 'read'
        ]
        
        # Visual-only keywords (appearance, layout, style)
        visual_keywords = [
            'look like', 'what color', 'what shape', 'where is', 'position',
            'layout', 'design', 'appearance', 'what does the diagram',
            'what does the figure', 'what does the image', 'what is shown',
            'what is depicted', 'what is illustrated', 'diagram show',
            'figure show', 'image show', 'what is the logo', 'what is the icon'
        ]
        
        # Hybrid keywords (need both text and visual understanding)
        hybrid_keywords = [
            'explain', 'how does', 'why does', 'what does this mean',
            'what is the relationship', 'compare', 'contrast', 'what trend',
            'what pattern', 'what conclusion', 'what inference', 'interpret',
            'what can you tell', 'what information', 'what insight'
        ]
        
        # Check text-only first
        for kw in text_keywords:
            if kw in q_lower:
                return 'text'
        
        # Check visual-only
        for kw in visual_keywords:
            if kw in q_lower:
                return 'visual'
        
        # Check hybrid
        for kw in hybrid_keywords:
            if kw in q_lower:
                return 'hybrid'
        
        # Default based on question length and structure
        if len(question.split()) < 8:
            return 'text'
        elif '?' in question and len(question.split()) > 10:
            return 'hybrid'
        
        return 'hybrid'  # default
    
    def _create_samples(self) -> List[DocVQASample]:
        """Convert annotations to sample objects"""
        samples = []
        
        # Handle different JSON structures
        if 'data' in self.annotations:
            items = self.annotations['data']
        else:
            items = self.annotations.get('annotations', [])
            if not items:
                # Try different key
                items = self.annotations
        
        for item in items:
            # Extract fields (handle different naming conventions)
            question_id = item.get('questionId', item.get('question_id', len(samples)))
            question = item.get('question', item.get('questionText', ''))
            image_id = item.get('imageId', item.get('image_id', item.get('docId', '')))
            answers = item.get('answers', item.get('answer', []))
            if isinstance(answers, str):
                answers = [answers]
            
            # Get question type from dataset if available
            dataset_qtype = item.get('questionType', item.get('type', None))
            
            # Classify question
            predicted_type = self._classify_question(question, dataset_qtype)
            
            # Build image path
            image_path = self.data_root / 'images' / f"{image_id}.jpg"
            if not image_path.exists():
                # Try .png
                image_path = self.data_root / 'images' / f"{image_id}.png"
            
            sample = DocVQASample(
                question_id=question_id,
                question=question,
                image_id=image_id,
                image_path=str(image_path),
                answers=answers,
                question_type=dataset_qtype,
                predicted_type=predicted_type,
                doc_id=image_id
            )
            samples.append(sample)
        
        return samples
    
    def _print_stats(self):
        """Print dataset statistics"""
        type_counts = {}
        for sample in self.samples:
            t = sample.predicted_type
            type_counts[t] = type_counts.get(t, 0) + 1
        
        logger.info(f"DocVQA {self.split} split:")
        logger.info(f"  Total samples: {len(self.samples)}")
        logger.info(f"  Text: {type_counts.get('text', 0)}")
        logger.info(f"  Visual: {type_counts.get('visual', 0)}")
        logger.info(f"  Hybrid: {type_counts.get('hybrid', 0)}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        return {
            'question_id': sample.question_id,
            'question': sample.question,
            'image_id': sample.image_id,
            'image_path': sample.image_path,
            'answers': sample.answers,
            'expected_type': sample.predicted_type,
            'doc_id': sample.doc_id,
            'split': self.split
        }
    
    def get_queries_for_routing(self) -> List[Tuple[str, str]]:
        """Return list of (expected_type, question) for router evaluation"""
        return [(s.predicted_type, s.question) for s in self.samples]


class RealInfoVQADataset:
    """
    Loader for Infographics VQA dataset (Task 3)
    """
    
    def __init__(self, data_root: str, split: str = 'train', max_samples: int = None):
        self.data_root = Path(data_root)
        self.split = split
        
        # Load annotations
        self.annotations = self._load_annotations()
        self.samples = self._create_samples()
        
        if max_samples:
            self.samples = self.samples[:max_samples]
        
        self._print_stats()
    
    def _load_annotations(self) -> Dict:
        """Load InfoVQA annotation JSON file"""
        ann_file = self.data_root / 'annotations' / f'{self.split}.json'
        
        if not ann_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {ann_file}")
        
        with open(ann_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded InfoVQA annotations from {ann_file}")
        return data
    
    def _classify_question(self, question: str) -> str:
        """Classify infoVQA questions"""
        q_lower = question.lower()
        
        # InfoVQA questions often require visual understanding of infographics
        # Most are hybrid (need to read text from infographics + understand visual layout)
        
        visual_keywords = ['color', 'position', 'located', 'where', 'icon', 'logo']
        text_keywords = ['what is the', 'what does the text', 'read', 'extract']
        
        if any(kw in q_lower for kw in visual_keywords):
            return 'visual'
        elif any(kw in q_lower for kw in text_keywords):
            return 'text'
        else:
            return 'hybrid'  # Most InfoVQA questions are hybrid
    
    def _create_samples(self) -> List:
        """Create sample objects"""
        samples = []
        
        items = self.annotations.get('data', self.annotations)
        
        for item in items:
            question = item.get('question', '')
            image_url = item.get('image_url', '')
            image_id = item.get('image_id', item.get('id', len(samples)))
            answers = item.get('answers', [])
            
            # Extract image filename from URL
            if image_url:
                image_name = image_url.split('/')[-1]
            else:
                image_name = f"{image_id}.jpg"
            
            image_path = self.data_root / 'images' / image_name
            
            predicted_type = self._classify_question(question)
            
            samples.append({
                'question_id': item.get('questionId', len(samples)),
                'question': question,
                'image_path': str(image_path),
                'answers': answers,
                'expected_type': predicted_type,
                'doc_id': image_id,
                'split': self.split
            })
        
        return samples
    
    def _print_stats(self):
        """Print statistics"""
        type_counts = {}
        for s in self.samples:
            t = s['expected_type']
            type_counts[t] = type_counts.get(t, 0) + 1
        
        logger.info(f"InfoVQA {self.split} split:")
        logger.info(f"  Total: {len(self.samples)}")
        logger.info(f"  Text: {type_counts.get('text', 0)}")
        logger.info(f"  Visual: {type_counts.get('visual', 0)}")
        logger.info(f"  Hybrid: {type_counts.get('hybrid', 0)}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]
    
    def get_queries_for_routing(self) -> List[Tuple[str, str]]:
        return [(s['expected_type'], s['question']) for s in self.samples]