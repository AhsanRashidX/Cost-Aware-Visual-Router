"""
Correct router model matching the checkpoint exactly
"""

import torch
import torch.nn as nn

class CorrectRouterModel(nn.Module):
    """
    Model architecture exactly matching the saved checkpoint
    """
    
    def __init__(self, input_dim=768, hidden_dim=256, num_classes=4, dropout=0.1):
        super().__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        shared_out = self.shared(x)
        logits = self.classifier(shared_out)
        return {'path_logits': logits, 'shared_features': shared_out}