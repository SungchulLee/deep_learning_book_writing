"""
Train Seq2Seq with Attention
"""
import torch
import torch.nn as nn
from seq2seq_with_attention import Seq2SeqWithAttention

# Simple training example
def train_simple_example():
    # Hyperparameters
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    embed_size = 256
    hidden_size = 512
    batch_size = 32
    
    model = Seq2SeqWithAttention(src_vocab_size, tgt_vocab_size, embed_size, hidden_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print("Training script ready!")
    print("\nTo train: Create your own dataset and training loop")

if __name__ == '__main__':
    train_simple_example()
