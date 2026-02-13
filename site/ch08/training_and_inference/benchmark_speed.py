"""
Benchmark Speed Comparison
"""
import torch
import time
from transformer_model import TransformerForComparison
from rnn_baseline import RNNBaseline
from cnn_baseline import CNNBaseline

def benchmark():
    batch_size = 32
    seq_len = 100
    input_dim = 64
    
    x = torch.randn(batch_size, seq_len, input_dim)
    
    models = {
        'Transformer': TransformerForComparison(input_dim),
        'RNN': RNNBaseline(input_dim),
        'CNN': CNNBaseline(input_channels=64)
    }
    
    for name, model in models.items():
        model.eval()
        start = time.time()
        with torch.no_grad():
            for _ in range(100):
                _ = model(x)
        elapsed = time.time() - start
        print(f"{name}: {elapsed:.3f}s for 100 forward passes")

if __name__ == '__main__':
    benchmark()
