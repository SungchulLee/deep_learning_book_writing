"""
Text Generation Sampling Strategies
"""
import torch
import torch.nn.functional as F

def greedy_sampling(logits):
    return torch.argmax(logits, dim=-1)

def top_k_sampling(logits, k=50, temperature=1.0):
    values, indices = torch.topk(logits / temperature, k)
    probs = F.softmax(values, dim=-1)
    next_token = torch.multinomial(probs, 1)
    return indices.gather(-1, next_token)

def nucleus_sampling(logits, p=0.9, temperature=1.0):
    sorted_logits, sorted_indices = torch.sort(logits / temperature, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = float('-inf')
    
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1)
