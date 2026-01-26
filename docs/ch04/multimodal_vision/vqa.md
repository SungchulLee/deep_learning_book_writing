# Visual Question Answering (VQA)

## Learning Objectives

By the end of this section, you will be able to:

1. Understand the VQA task formulation and its unique challenges
2. Design multimodal fusion architectures for joint reasoning
3. Implement attention mechanisms for question-guided visual understanding
4. Apply different answer prediction strategies
5. Evaluate VQA systems using standard metrics

## Introduction

Visual Question Answering (VQA) is the task of answering natural language questions about images. It requires understanding specific aspects of an image based on the question, demanding both visual recognition and language comprehension skills.

## Problem Formulation

Given an image $I$ and a question $Q$, predict the answer $A$:

$$A^* = \arg\max_A P(A | I, Q)$$

## Architecture Overview

```
Image ──→ [Visual Encoder] ──→ Visual Features (V)
                                      ↓
                             [Multimodal Fusion]
                                      ↑
Question ──→ [Text Encoder] ──→ Question Features (Q)
                                      ↓
                             [Answer Predictor] ──→ Answer
```

## PyTorch Implementation

### Stacked Attention Network

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class StackedAttention(nn.Module):
    """
    Stacked Attention Network for VQA.
    Multiple rounds of attention progressively refine visual understanding.
    """
    
    def __init__(self, visual_dim: int, question_dim: int,
                 attention_dim: int = 512, num_stacks: int = 2):
        super().__init__()
        self.num_stacks = num_stacks
        
        self.attention_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(visual_dim + question_dim, attention_dim),
                nn.Tanh(),
                nn.Dropout(0.2),
                nn.Linear(attention_dim, 1)
            ) for _ in range(num_stacks)
        ])
        
        self.query_updates = nn.ModuleList([
            nn.Linear(visual_dim + question_dim, question_dim)
            for _ in range(num_stacks)
        ])
        
    def forward(self, visual_features: torch.Tensor,
               question_feature: torch.Tensor) -> Tuple[torch.Tensor, list]:
        batch_size, num_regions, _ = visual_features.shape
        query = question_feature
        all_weights = []
        
        for i in range(self.num_stacks):
            query_exp = query.unsqueeze(1).expand(-1, num_regions, -1)
            combined = torch.cat([visual_features, query_exp], dim=-1)
            
            scores = self.attention_layers[i](combined).squeeze(-1)
            weights = F.softmax(scores, dim=-1)
            all_weights.append(weights)
            
            attended = (visual_features * weights.unsqueeze(-1)).sum(dim=1)
            query = query + self.query_updates[i](torch.cat([attended, query], dim=-1))
        
        return torch.cat([query, attended], dim=-1), all_weights


class VQAModel(nn.Module):
    """Complete VQA model with attention-based fusion."""
    
    def __init__(self, vocab_size: int, num_answers: int,
                 embed_dim: int = 300, hidden_dim: int = 512,
                 visual_dim: int = 2048):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.question_lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.question_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.attention = StackedAttention(hidden_dim, hidden_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_answers)
        )
    
    def forward(self, image_features: torch.Tensor, 
                questions: torch.Tensor) -> torch.Tensor:
        # Encode question
        embedded = self.embedding(questions)
        _, (h_n, _) = self.question_lstm(embedded)
        q_feat = self.question_proj(torch.cat([h_n[-2], h_n[-1]], dim=-1))
        
        # Project visual features
        v_feat = self.visual_proj(image_features)
        
        # Attention fusion
        fused, _ = self.attention(v_feat, q_feat)
        
        # Classify answer
        return self.classifier(fused)
```

## Training and Evaluation

```python
def train_vqa(model, train_loader, num_epochs=20, lr=1e-4, device='cuda'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for images, questions, answers in train_loader:
            images, questions, answers = images.to(device), questions.to(device), answers.to(device)
            
            optimizer.zero_grad()
            logits = model(images, questions)
            loss = criterion(logits, answers)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
```

## Key Datasets

| Dataset | Images | Questions | Answer Type |
|---------|--------|-----------|-------------|
| VQA v2 | 204K | 1.1M | Open-ended |
| GQA | 113K | 22M | Compositional |
| Visual Genome | 108K | 1.7M | Dense annotations |

## References

1. Antol, S., et al. "VQA: Visual Question Answering." ICCV 2015.
2. Yang, Z., et al. "Stacked Attention Networks for Image Question Answering." CVPR 2016.
3. Anderson, P., et al. "Bottom-Up and Top-Down Attention for Image Captioning and VQA." CVPR 2018.
