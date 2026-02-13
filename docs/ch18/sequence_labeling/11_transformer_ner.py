"""
Transformer-based NER using Pre-trained Models
===============================================

Using BERT/RoBERTa for Named Entity Recognition.

Key concepts:
- Fine-tuning pre-trained transformers
- Subword tokenization handling
- Token classification

Author: Educational purposes
Date: 2025
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict


class TransformerNER(nn.Module):
    """
    Transformer-based NER using pre-trained models.
    
    Architecture:
    Input → BERT/RoBERTa → Linear → Softmax → Tags
    """
    
    def __init__(self, model_name: str = 'bert-base-cased', num_labels: int = 9):
        """
        Initialize Transformer NER.
        
        Args:
            model_name: Hugging Face model name
            num_labels: Number of entity labels
        """
        super(TransformerNER, self).__init__()
        
        self.num_labels = num_labels
        self.model_name = model_name
        
        # Load pre-trained transformer
        self.transformer = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
        # Classification head
        hidden_size = self.transformer.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass.
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            
        Returns:
            logits: [batch_size, seq_len, num_labels]
        """
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get hidden states
        sequence_output = outputs[0]  # [batch, seq_len, hidden_size]
        
        # Apply dropout
        sequence_output = self.dropout(sequence_output)
        
        # Classification
        logits = self.classifier(sequence_output)  # [batch, seq_len, num_labels]
        
        return logits
    
    def predict(self, text: str) -> List[Tuple[str, str]]:
        """
        Predict entities in text.
        
        Args:
            text: Input text string
            
        Returns:
            List of (token, label) pairs
        """
        # Tokenize
        encoding = self.tokenizer(text, return_tensors='pt', 
                                  padding=True, truncation=True)
        
        # Forward pass
        with torch.no_grad():
            logits = self.forward(encoding['input_ids'], 
                                 encoding['attention_mask'])
        
        # Get predictions
        predictions = torch.argmax(logits, dim=-1)[0]
        
        # Map back to tokens
        tokens = self.tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
        
        return list(zip(tokens, predictions.tolist()))


if __name__ == "__main__":
    print("Transformer NER model template")
    print("Note: Requires transformers library and pre-trained models")
    print("Example: BERT, RoBERTa, DistilBERT for token classification")
