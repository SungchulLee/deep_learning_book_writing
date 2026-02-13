"""
Beginner Level: Self-Attention Heatmap with Real Models
========================================================

This module demonstrates how to extract and visualize attention weights from
pre-trained transformer models (BERT, GPT-2, etc.). It builds on the basics
by working with actual models rather than synthetic data.

Learning Goals:
--------------
1. Load and use pre-trained transformer models
2. Extract attention weights from real models
3. Visualize attention for actual text inputs
4. Compare attention across different layers
5. Understand attention in masked vs. bidirectional models

Mathematical Concepts:
---------------------
Self-attention computes relationships between all positions in a sequence:
    
    For each position i:
        Q_i = x_i @ W_Q  (query)
        K_j = x_j @ W_K  (key) for all j
        V_j = x_j @ W_V  (value) for all j
    
    attention_i = softmax(Q_i @ K^T / sqrt(d_k))
    output_i = attention_i @ V

The attention weights matrix A where A[i,j] = attention_i[j] shows
how much position i attends to position j.

Author: Deep Learning Curriculum
Date: November 2025
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# Note: These imports require transformers library
# Install with: pip install transformers --break-system-packages
try:
    from transformers import (
        BertTokenizer, BertModel,
        GPT2Tokenizer, GPT2LMHeadModel,
        AutoTokenizer, AutoModel
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers library not installed. Using mock data.")


class TransformerAttentionVisualizer:
    """
    Visualizer for extracting and displaying attention from pre-trained transformers.
    
    This class handles the specifics of working with Hugging Face transformer models,
    including tokenization, model loading, and attention extraction.
    
    Supported Models:
    ----------------
    - BERT (bert-base-uncased, bert-large-uncased)
    - GPT-2 (gpt2, gpt2-medium, gpt2-large)
    - DistilBERT (distilbert-base-uncased)
    - RoBERTa (roberta-base)
    - Any model from Hugging Face that supports output_attentions=True
    
    Attributes:
    ----------
    model_name : str
        Name of the pre-trained model
    model : nn.Module
        The loaded transformer model
    tokenizer : PreTrainedTokenizer
        The model's tokenizer
    device : str
        Device to run model on ('cuda' or 'cpu')
    """
    
    def __init__(self, model_name: str = "bert-base-uncased", device: Optional[str] = None):
        """
        Initialize the visualizer with a pre-trained model.
        
        Parameters:
        ----------
        model_name : str, default="bert-base-uncased"
            Name of the Hugging Face model to load
        device : str, optional
            Device to use ('cuda' or 'cpu'). If None, auto-detect.
            
        Examples:
        --------
        >>> # Load BERT model
        >>> viz = TransformerAttentionVisualizer("bert-base-uncased")
        
        >>> # Load GPT-2 model
        >>> viz = TransformerAttentionVisualizer("gpt2")
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library is required. "
                "Install with: pip install transformers --break-system-packages"
            )
        
        self.model_name = model_name
        
        # Detect device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"Loading model: {model_name}")
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        # AutoTokenizer and AutoModel work for most Hugging Face models
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, output_attentions=True)
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        print(f"Model loaded successfully!")
        print(f"Number of layers: {self.model.config.num_hidden_layers}")
        print(f"Number of attention heads: {self.model.config.num_attention_heads}")
    
    def get_attention_weights(self,
                            text: str,
                            layer_idx: int = -1,
                            head_idx: int = 0) -> Tuple[torch.Tensor, List[str]]:
        """
        Get attention weights for a text input.
        
        This method:
        1. Tokenizes the input text
        2. Runs forward pass through the model
        3. Extracts attention weights from specified layer and head
        4. Returns weights and corresponding tokens
        
        Parameters:
        ----------
        text : str
            Input text to analyze
        layer_idx : int, default=-1
            Layer to extract attention from (-1 for last layer)
        head_idx : int, default=0
            Attention head to extract (0-indexed)
        
        Returns:
        -------
        attention_weights : torch.Tensor
            Attention weight matrix, shape: (seq_len, seq_len)
        tokens : list of str
            List of tokens (including special tokens like [CLS], [SEP])
            
        Notes:
        -----
        - BERT uses [CLS] at start and [SEP] at end
        - GPT-2 uses <|endoftext|> tokens
        - Attention includes these special tokens
        """
        # Tokenize input
        # return_tensors='pt' gives PyTorch tensors
        # padding=True adds padding if needed
        # truncation=True truncates if too long
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512  # Most models have 512 max length
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Extract attention weights
        # outputs.attentions is a tuple of length num_layers
        # Each element: (batch_size, num_heads, seq_len, seq_len)
        attentions = outputs.attentions
        
        # Get specific layer (handle negative indexing)
        if layer_idx < 0:
            layer_idx = len(attentions) + layer_idx
        
        # Extract attention: (batch_size=1, num_heads, seq_len, seq_len)
        layer_attention = attentions[layer_idx]
        
        # Get specific head: (seq_len, seq_len)
        attention_weights = layer_attention[0, head_idx].cpu()
        
        # Get tokens (convert token IDs back to strings)
        input_ids = inputs['input_ids'][0].cpu().numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        
        return attention_weights, tokens
    
    def plot_attention_heatmap(self,
                              text: str,
                              layer_idx: int = -1,
                              head_idx: int = 0,
                              save_path: Optional[str] = None) -> None:
        """
        Plot attention heatmap for given text.
        
        Parameters:
        ----------
        text : str
            Input text to visualize
        layer_idx : int, default=-1
            Layer to visualize (-1 for last layer)
        head_idx : int, default=0
            Attention head to visualize
        save_path : str, optional
            Path to save figure
        """
        # Get attention weights and tokens
        attention, tokens = self.get_attention_weights(text, layer_idx, head_idx)
        
        # Convert to numpy
        attention = attention.numpy()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create heatmap
        sns.heatmap(
            attention,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap='viridis',
            square=True,
            cbar_kws={'label': 'Attention Weight'},
            ax=ax,
            vmin=0,
            vmax=1
        )
        
        # Labels and title
        ax.set_xlabel('Key Tokens', fontsize=12, fontweight='bold')
        ax.set_ylabel('Query Tokens', fontsize=12, fontweight='bold')
        
        # Create informative title
        title = (f"Attention Heatmap\n"
                f"Model: {self.model_name} | "
                f"Layer: {layer_idx if layer_idx >= 0 else len(self.model.encoder.layer) + layer_idx} | "
                f"Head: {head_idx}")
        ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
        
        # Rotate labels
        plt.xticks(rotation=90, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"Saved to: {save_path}")
        
        plt.show()
    
    def compare_layers(self,
                      text: str,
                      layers: List[int],
                      head_idx: int = 0,
                      save_path: Optional[str] = None) -> None:
        """
        Compare attention patterns across multiple layers.
        
        This helps understand how attention evolves through the network:
        - Early layers: Often focus on local/syntactic patterns
        - Middle layers: Capture more semantic relationships
        - Late layers: Task-specific attention
        
        Parameters:
        ----------
        text : str
            Input text to analyze
        layers : list of int
            List of layer indices to compare
        head_idx : int, default=0
            Which attention head to visualize
        save_path : str, optional
            Path to save figure
        """
        num_layers = len(layers)
        
        # Create subplot grid
        fig, axes = plt.subplots(1, num_layers, figsize=(6 * num_layers, 5))
        if num_layers == 1:
            axes = [axes]
        
        # Get tokens once (same for all layers)
        _, tokens = self.get_attention_weights(text, layers[0], head_idx)
        
        # Plot each layer
        for idx, (layer_idx, ax) in enumerate(zip(layers, axes)):
            # Get attention for this layer
            attention, _ = self.get_attention_weights(text, layer_idx, head_idx)
            attention = attention.numpy()
            
            # Create heatmap
            sns.heatmap(
                attention,
                xticklabels=tokens if idx == 0 else [],  # Only show labels on first plot
                yticklabels=tokens,
                cmap='viridis',
                square=True,
                cbar=True,
                ax=ax,
                vmin=0,
                vmax=1,
                cbar_kws={'label': 'Attention'} if idx == num_layers - 1 else {}
            )
            
            ax.set_title(f'Layer {layer_idx}', fontsize=12, fontweight='bold')
            
            if idx == 0:
                ax.set_ylabel('Query Tokens', fontsize=10, fontweight='bold')
            
            # Rotate labels
            if idx == 0:
                ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right')
        
        # Overall title
        fig.suptitle(
            f'Attention Across Layers - Model: {self.model_name}, Head: {head_idx}',
            fontsize=14,
            fontweight='bold',
            y=1.02
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"Saved to: {save_path}")
        
        plt.show()
    
    def compare_heads(self,
                     text: str,
                     layer_idx: int = -1,
                     num_heads: Optional[int] = None,
                     save_path: Optional[str] = None) -> None:
        """
        Compare attention patterns across multiple heads.
        
        Different attention heads often specialize in different patterns:
        - Some focus on positional information
        - Some capture syntactic relationships
        - Some learn semantic associations
        
        Parameters:
        ----------
        text : str
            Input text to analyze
        layer_idx : int, default=-1
            Which layer to visualize
        num_heads : int, optional
            Number of heads to show (default: show all or first 6)
        save_path : str, optional
            Path to save figure
        """
        # Determine number of heads to show
        total_heads = self.model.config.num_attention_heads
        if num_heads is None:
            num_heads = min(6, total_heads)  # Show max 6 for readability
        
        # Create subplot grid (2 rows)
        ncols = min(3, num_heads)
        nrows = (num_heads + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = axes.flatten() if num_heads > 1 else [axes]
        
        # Get tokens once
        _, tokens = self.get_attention_weights(text, layer_idx, 0)
        
        # Plot each head
        for head_idx in range(num_heads):
            ax = axes[head_idx]
            
            # Get attention for this head
            attention, _ = self.get_attention_weights(text, layer_idx, head_idx)
            attention = attention.numpy()
            
            # Create heatmap
            sns.heatmap(
                attention,
                xticklabels=[],  # Hide labels for cleaner look
                yticklabels=tokens if head_idx % ncols == 0 else [],
                cmap='viridis',
                square=True,
                cbar=True,
                ax=ax,
                vmin=0,
                vmax=1,
                cbar_kws={'label': 'Attention', 'shrink': 0.8}
            )
            
            ax.set_title(f'Head {head_idx}', fontsize=11, fontweight='bold')
            
            if head_idx % ncols == 0:
                ax.set_ylabel('Query Tokens', fontsize=9)
        
        # Hide extra subplots
        for idx in range(num_heads, len(axes)):
            axes[idx].axis('off')
        
        # Overall title
        fig.suptitle(
            f'Multi-Head Attention Comparison\n'
            f'Model: {self.model_name} | Layer: {layer_idx}',
            fontsize=14,
            fontweight='bold'
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"Saved to: {save_path}")
        
        plt.show()
    
    def analyze_token_attention(self,
                               text: str,
                               focus_token: str,
                               layer_idx: int = -1,
                               head_idx: int = 0) -> None:
        """
        Analyze how a specific token attends to others.
        
        Parameters:
        ----------
        text : str
            Input text
        focus_token : str
            Token to focus on (must be in text)
        layer_idx : int, default=-1
            Layer to analyze
        head_idx : int, default=0
            Head to analyze
        """
        # Get attention and tokens
        attention, tokens = self.get_attention_weights(text, layer_idx, head_idx)
        attention = attention.numpy()
        
        # Find token index
        # Note: Tokenization might split words, so we find the closest match
        token_idx = None
        for idx, token in enumerate(tokens):
            if focus_token.lower() in token.lower():
                token_idx = idx
                break
        
        if token_idx is None:
            print(f"Token '{focus_token}' not found in sequence.")
            print(f"Available tokens: {tokens}")
            return
        
        # Extract attention weights for this token
        token_attention = attention[token_idx, :]
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(14, 6))
        
        positions = np.arange(len(tokens))
        colors = ['steelblue'] * len(tokens)
        colors[token_idx] = 'coral'  # Highlight focus token
        
        bars = ax.bar(positions, token_attention, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels for top 5 attentions
        top_5_indices = np.argsort(token_attention)[-5:]
        for idx in top_5_indices:
            ax.text(idx, token_attention[idx] + 0.01, f'{token_attention[idx]:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Customize
        ax.set_xlabel('Tokens', fontsize=12, fontweight='bold')
        ax.set_ylabel('Attention Weight', fontsize=12, fontweight='bold')
        ax.set_title(
            f"Attention from '{tokens[token_idx]}' to all tokens\n"
            f"Model: {self.model_name} | Layer: {layer_idx} | Head: {head_idx}",
            fontsize=13,
            fontweight='bold',
            pad=20
        )
        
        ax.set_xticks(positions)
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        
        # Add uniform baseline
        uniform = 1.0 / len(tokens)
        ax.axhline(uniform, color='red', linestyle='--', 
                  label=f'Uniform: {uniform:.3f}', linewidth=1.5)
        
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()
        
        plt.tight_layout()
        plt.show()


# ============================================================================
# Example Usage with Real Transformer Models
# ============================================================================

def example_bert_attention():
    """
    Example 1: Basic BERT attention visualization.
    
    Demonstrates:
    - Loading BERT model
    - Extracting attention from last layer
    - Visualizing with heatmap
    """
    print("=" * 70)
    print("Example 1: BERT Self-Attention Visualization")
    print("=" * 70)
    
    if not TRANSFORMERS_AVAILABLE:
        print("Transformers library not available. Skipping example.")
        return
    
    # Initialize visualizer with BERT
    viz = TransformerAttentionVisualizer("bert-base-uncased")
    
    # Sample text
    text = "The quick brown fox jumps over the lazy dog"
    
    print(f"\nAnalyzing text: '{text}'")
    
    # Visualize attention from last layer, first head
    viz.plot_attention_heatmap(
        text,
        layer_idx=-1,
        head_idx=0
    )


def example_layer_comparison():
    """
    Example 2: Compare attention across layers.
    
    Demonstrates:
    - Evolution of attention through layers
    - Early vs late layer patterns
    """
    print("\n" + "=" * 70)
    print("Example 2: Attention Evolution Across Layers")
    print("=" * 70)
    
    if not TRANSFORMERS_AVAILABLE:
        print("Transformers library not available. Skipping example.")
        return
    
    viz = TransformerAttentionVisualizer("bert-base-uncased")
    
    text = "I love machine learning and deep neural networks"
    
    print(f"\nAnalyzing text: '{text}'")
    print("Comparing layers: 0 (early), 5 (middle), 11 (late)")
    
    # Compare first, middle, and last layers
    viz.compare_layers(
        text,
        layers=[0, 5, 11],
        head_idx=0
    )


def example_multihead_comparison():
    """
    Example 3: Compare different attention heads.
    
    Demonstrates:
    - Multi-head attention diversity
    - Different heads learning different patterns
    """
    print("\n" + "=" * 70)
    print("Example 3: Multi-Head Attention Comparison")
    print("=" * 70)
    
    if not TRANSFORMERS_AVAILABLE:
        print("Transformers library not available. Skipping example.")
        return
    
    viz = TransformerAttentionVisualizer("bert-base-uncased")
    
    text = "Attention mechanisms are crucial for transformers"
    
    print(f"\nAnalyzing text: '{text}'")
    print("Comparing first 6 attention heads in last layer")
    
    # Compare multiple heads
    viz.compare_heads(
        text,
        layer_idx=-1,
        num_heads=6
    )


def example_token_analysis():
    """
    Example 4: Analyze specific token attention.
    
    Demonstrates:
    - Focusing on individual tokens
    - Understanding token relationships
    """
    print("\n" + "=" * 70)
    print("Example 4: Token-Specific Attention Analysis")
    print("=" * 70)
    
    if not TRANSFORMERS_AVAILABLE:
        print("Transformers library not available. Skipping example.")
        return
    
    viz = TransformerAttentionVisualizer("bert-base-uncased")
    
    text = "The cat sat on the mat and looked at the rat"
    
    print(f"\nAnalyzing text: '{text}'")
    
    # Analyze attention from specific tokens
    focus_tokens = ["cat", "mat", "rat"]
    
    for token in focus_tokens:
        print(f"\nAnalyzing attention from: '{token}'")
        viz.analyze_token_attention(
            text,
            focus_token=token,
            layer_idx=-1,
            head_idx=0
        )


def example_sentence_comparison():
    """
    Example 5: Compare attention for different sentences.
    
    Demonstrates:
    - How sentence structure affects attention
    - Comparing similar vs different sentences
    """
    print("\n" + "=" * 70)
    print("Example 5: Sentence Comparison")
    print("=" * 70)
    
    if not TRANSFORMERS_AVAILABLE:
        print("Transformers library not available. Skipping example.")
        return
    
    viz = TransformerAttentionVisualizer("bert-base-uncased")
    
    sentences = [
        "The dog chased the cat",
        "The cat was chased by the dog",
        "Dogs and cats are pets"
    ]
    
    print("\nComparing attention patterns for different sentences:")
    
    for sent in sentences:
        print(f"\n  - {sent}")
        viz.plot_attention_heatmap(
            sent,
            layer_idx=-1,
            head_idx=0
        )


if __name__ == "__main__":
    """
    Main execution with comprehensive examples.
    
    Run this script to see attention visualization with real models:
        python self_attention_heatmap.py
    
    Note: First run will download model weights (~400MB for BERT-base)
    """
    
    print("\n" + "=" * 70)
    print(" SELF-ATTENTION HEATMAP WITH REAL TRANSFORMERS ")
    print(" Module 59: Working with Pre-trained Models ")
    print("=" * 70)
    
    if not TRANSFORMERS_AVAILABLE:
        print("\nError: transformers library not installed.")
        print("Install with: pip install transformers --break-system-packages")
        print("\nThis script requires real transformer models.")
        exit(1)
    
    # Run examples
    # Comment out examples you don't want to run
    
    example_bert_attention()
    # example_layer_comparison()  # Uncomment to run
    # example_multihead_comparison()  # Uncomment to run
    # example_token_analysis()  # Uncomment to run
    # example_sentence_comparison()  # Uncomment to run
    
    print("\n" + "=" * 70)
    print(" Examples completed! ")
    print("=" * 70)
    print("\nKey Insights:")
    print("  1. BERT uses bidirectional attention (all tokens can attend to all)")
    print("  2. Early layers show more local patterns")
    print("  3. Late layers capture more semantic relationships")
    print("  4. Different heads specialize in different patterns")
    print("  5. Special tokens ([CLS], [SEP]) have unique attention patterns")
