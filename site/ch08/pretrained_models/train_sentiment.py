"""
Train BERT for Sentiment Analysis
"""
import torch
import torch.nn as nn
from bert_classifier import BERTClassifier

def train_sentiment():
    vocab_size = 10000
    num_classes = 2
    model = BERTClassifier(vocab_size, num_classes)
    
    print(f"BERT Classifier created!")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Ready for training on sentiment analysis tasks")

if __name__ == '__main__':
    train_sentiment()
