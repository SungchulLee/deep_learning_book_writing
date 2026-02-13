"""
Module 35: Multimodal Vision - Advanced Level
==============================================

Topic: Vision-Language Transformers with Cross-Modal Attention

This script implements a full vision-language transformer with:
- Cross-modal attention mechanisms
- Multi-task pretraining (ITM + MLM + Contrastive)
- Multimodal fusion strategies
- Applications: Image captioning, Visual Question Answering (VQA)

Key Concepts:
1. Cross-attention between vision and language modalities
2. Multimodal fusion: early vs late vs intermediate
3. Multiple pretraining objectives
4. Vision-language transformer architecture (ViLBERT/ALBEF-style)
5. Task-specific fine-tuning

Learning Objectives:
- Build cross-modal attention layers
- Implement multi-task training objectives
- Design multimodal fusion strategies
- Apply to image captioning and VQA
- Understand vision-language pretraining

Mathematical Background:
- Cross-Attention: Q(text) attends to K,V(image)
- Self-Attention: Within-modality attention
- ITM: Binary classification P(match|image,text)
- MLM: Masked language modeling P(word|context,image)

Author: Educational AI
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, List, Optional, Dict
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

# Set seeds
torch.manual_seed(42)
np.random.seed(42)


# ============================================================================
# PART 1: MULTI-HEAD ATTENTION COMPONENTS
# ============================================================================

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.
    
    Attention computes weighted average of values based on query-key similarity:
        Attention(Q, K, V) = softmax(QK^T / √d_k) V
    
    Multi-head attention runs multiple attention operations in parallel:
    - Different heads can attend to different aspects of the input
    - Heads are concatenated and linearly transformed
    
    Mathematical Details:
        head_i = Attention(Q·W_Q^i, K·W_K^i, V·W_V^i)
        MultiHead(Q,K,V) = Concat(head_1,...,head_h)·W_O
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Linear projections for Q, K, V
        # We use one projection for all heads (more efficient)
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout for attention weights
        self.dropout = nn.Dropout(dropout)
        
        # Scale factor for attention scores
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for multi-head attention.
        
        Args:
            query: Query tensor (batch, query_len, embed_dim)
            key: Key tensor (batch, key_len, embed_dim)
            value: Value tensor (batch, key_len, embed_dim)
            attention_mask: Optional mask (batch, query_len, key_len)
                          1 = attend, 0 = mask out
        
        Returns:
            Tuple of (output, attention_weights)
            output: (batch, query_len, embed_dim)
            attention_weights: (batch, num_heads, query_len, key_len)
        """
        batch_size = query.shape[0]
        query_len = query.shape[1]
        key_len = key.shape[1]
        
        # Linear projections
        # Shape: (batch, seq_len, embed_dim)
        Q = self.query_proj(query)
        K = self.key_proj(key)
        V = self.value_proj(value)
        
        # Reshape for multi-head attention
        # Shape: (batch, seq_len, num_heads, head_dim)
        # Then transpose to: (batch, num_heads, seq_len, head_dim)
        Q = Q.view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        # QK^T: (batch, num_heads, query_len, head_dim) @ (batch, num_heads, head_dim, key_len)
        # Result: (batch, num_heads, query_len, key_len)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask to match attention_scores shape
            # attention_mask: (batch, query_len, key_len) -> (batch, 1, query_len, key_len)
            attention_mask = attention_mask.unsqueeze(1)
            # Mask out by setting to large negative value (will be ~0 after softmax)
            attention_scores = attention_scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Apply softmax to get attention weights
        # Shape: (batch, num_heads, query_len, key_len)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        # (batch, num_heads, query_len, key_len) @ (batch, num_heads, key_len, head_dim)
        # Result: (batch, num_heads, query_len, head_dim)
        context = torch.matmul(attention_weights, V)
        
        # Reshape back
        # Transpose: (batch, query_len, num_heads, head_dim)
        # Reshape: (batch, query_len, embed_dim)
        context = context.transpose(1, 2).contiguous().view(batch_size, query_len, self.embed_dim)
        
        # Final output projection
        output = self.out_proj(context)
        
        return output, attention_weights


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention layer for vision-language interaction.
    
    This allows one modality (query) to attend to another modality (key/value).
    For example:
    - Text queries attend to image features (text-to-vision)
    - Image queries attend to text features (vision-to-text)
    
    Architecture:
    1. Multi-head cross-attention
    2. Add & Norm (residual connection + layer norm)
    3. Feed-forward network
    4. Add & Norm
    
    This is the key component that enables multimodal understanding!
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        ffn_dim: Feed-forward network hidden dimension
        dropout: Dropout probability
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 8, 
                 ffn_dim: int = 2048, dropout: float = 0.1):
        super(CrossModalAttention, self).__init__()
        
        # Cross-attention layer
        self.cross_attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),  # GELU activation (smoother than ReLU)
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, query_modality: torch.Tensor, 
                key_value_modality: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for cross-modal attention.
        
        Args:
            query_modality: Query features (batch, query_len, embed_dim)
            key_value_modality: Key/Value features (batch, kv_len, embed_dim)
            attention_mask: Optional mask (batch, query_len, kv_len)
        
        Returns:
            Output features (batch, query_len, embed_dim)
        """
        # Cross-attention with residual connection
        # Query from one modality, Key/Value from another
        attn_output, _ = self.cross_attention(
            query=query_modality,
            key=key_value_modality,
            value=key_value_modality,
            attention_mask=attention_mask
        )
        
        # Add & Norm (residual connection)
        query_modality = self.norm1(query_modality + attn_output)
        
        # Feed-forward network with residual
        ffn_output = self.ffn(query_modality)
        output = self.norm2(query_modality + ffn_output)
        
        return output


# ============================================================================
# PART 2: VISION-LANGUAGE TRANSFORMER ENCODER
# ============================================================================

class TransformerEncoderLayer(nn.Module):
    """
    Standard Transformer encoder layer with self-attention.
    
    Used for within-modality processing (before cross-modal fusion).
    
    Architecture:
    1. Multi-head self-attention
    2. Add & Norm
    3. Feed-forward network
    4. Add & Norm
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 8,
                 ffn_dim: int = 2048, dropout: float = 0.1):
        super(TransformerEncoderLayer, self).__init__()
        
        self.self_attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for transformer encoder layer.
        
        Args:
            x: Input features (batch, seq_len, embed_dim)
            attention_mask: Optional mask (batch, seq_len, seq_len)
        
        Returns:
            Output features (batch, seq_len, embed_dim)
        """
        # Self-attention with residual
        attn_output, _ = self.self_attention(x, x, x, attention_mask)
        x = self.norm1(x + attn_output)
        
        # FFN with residual
        ffn_output = self.ffn(x)
        output = self.norm2(x + ffn_output)
        
        return output


class VisionLanguageTransformer(nn.Module):
    """
    Complete Vision-Language Transformer with cross-modal fusion.
    
    Architecture (ALBEF/ViLBERT-style):
    1. Separate encoders for vision and text (self-attention)
    2. Cross-modal layers for vision-language interaction
    3. Multi-task pretraining objectives:
       - Image-Text Contrastive (ITC): Like CLIP
       - Image-Text Matching (ITM): Binary classification
       - Masked Language Modeling (MLM): Predict masked words
    
    This architecture allows for rich multimodal understanding by enabling
    vision and language to interact through cross-attention.
    
    Args:
        image_dim: Input image feature dimension
        text_dim: Input text feature dimension
        embed_dim: Hidden/embedding dimension
        num_layers_single: Number of single-modal encoder layers
        num_layers_cross: Number of cross-modal layers
        num_heads: Number of attention heads
        ffn_dim: Feed-forward network dimension
        vocab_size: Vocabulary size for MLM
        max_text_len: Maximum text sequence length
    """
    
    def __init__(self, image_dim: int = 2048, text_dim: int = 768,
                 embed_dim: int = 512, num_layers_single: int = 2,
                 num_layers_cross: int = 2, num_heads: int = 8,
                 ffn_dim: int = 2048, vocab_size: int = 30000,
                 max_text_len: int = 50, dropout: float = 0.1):
        super(VisionLanguageTransformer, self).__init__()
        
        self.embed_dim = embed_dim
        self.max_text_len = max_text_len
        
        # ========== Input Projections ==========
        # Project image features to embed_dim
        self.image_proj = nn.Linear(image_dim, embed_dim)
        
        # Project text features to embed_dim
        self.text_proj = nn.Linear(text_dim, embed_dim)
        
        # Positional embeddings for text
        self.text_pos_embed = nn.Parameter(torch.randn(1, max_text_len, embed_dim))
        
        # ========== Single-Modal Encoders ==========
        # Vision encoder (self-attention within images)
        self.vision_encoder = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers_single)
        ])
        
        # Text encoder (self-attention within text)
        self.text_encoder = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers_single)
        ])
        
        # ========== Cross-Modal Layers ==========
        # Vision-to-Text cross-attention (text queries attend to vision)
        self.v2t_cross_layers = nn.ModuleList([
            CrossModalAttention(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers_cross)
        ])
        
        # Text-to-Vision cross-attention (vision queries attend to text)
        self.t2v_cross_layers = nn.ModuleList([
            CrossModalAttention(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers_cross)
        ])
        
        # ========== Task-Specific Heads ==========
        
        # Image-Text Matching head (binary classification)
        # Predicts if (image, text) pair is matched or not
        self.itm_head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 2)  # Binary: match or not
        )
        
        # Masked Language Modeling head
        # Predicts masked words given image and text context
        self.mlm_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, vocab_size)
        )
        
        # Contrastive projection head (for ITC)
        # Projects to space for contrastive learning
        self.vision_proj_head = nn.Linear(embed_dim, embed_dim)
        self.text_proj_head = nn.Linear(embed_dim, embed_dim)
        
        # Temperature for contrastive learning
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        self.dropout = nn.Dropout(dropout)
        
    def encode_vision(self, image_features: torch.Tensor) -> torch.Tensor:
        """
        Encode image features through vision encoder.
        
        Args:
            image_features: (batch, image_dim)
        
        Returns:
            Vision embeddings (batch, 1, embed_dim)
        """
        # Project to embedding dimension
        # Shape: (batch, embed_dim)
        vision_emb = self.image_proj(image_features)
        
        # Add batch dimension: (batch, 1, embed_dim)
        # We treat image as single token for simplicity
        vision_emb = vision_emb.unsqueeze(1)
        
        # Apply dropout
        vision_emb = self.dropout(vision_emb)
        
        # Pass through vision encoder layers (self-attention)
        for layer in self.vision_encoder:
            vision_emb = layer(vision_emb)
        
        return vision_emb
    
    def encode_text(self, text_features: torch.Tensor) -> torch.Tensor:
        """
        Encode text features through text encoder.
        
        Args:
            text_features: (batch, seq_len, text_dim)
        
        Returns:
            Text embeddings (batch, seq_len, embed_dim)
        """
        batch_size, seq_len = text_features.shape[:2]
        
        # Project to embedding dimension
        # Shape: (batch, seq_len, embed_dim)
        text_emb = self.text_proj(text_features)
        
        # Add positional embeddings
        # Helps model understand word order
        text_emb = text_emb + self.text_pos_embed[:, :seq_len, :]
        
        # Apply dropout
        text_emb = self.dropout(text_emb)
        
        # Pass through text encoder layers (self-attention)
        for layer in self.text_encoder:
            text_emb = layer(text_emb)
        
        return text_emb
    
    def cross_modal_fusion(self, vision_emb: torch.Tensor, 
                          text_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse vision and text through cross-modal attention.
        
        This is where the magic happens! Vision and text interact:
        - Text queries attend to vision (understand visual content)
        - Vision queries attend to text (ground visual features in language)
        
        Args:
            vision_emb: Vision embeddings (batch, 1, embed_dim)
            text_emb: Text embeddings (batch, seq_len, embed_dim)
        
        Returns:
            Tuple of (fused_vision, fused_text)
        """
        # Alternate between V2T and T2V cross-attention
        for v2t_layer, t2v_layer in zip(self.v2t_cross_layers, self.t2v_cross_layers):
            # Text attends to vision (text queries, vision key/value)
            text_emb = v2t_layer(
                query_modality=text_emb,
                key_value_modality=vision_emb
            )
            
            # Vision attends to text (vision queries, text key/value)
            vision_emb = t2v_layer(
                query_modality=vision_emb,
                key_value_modality=text_emb
            )
        
        return vision_emb, text_emb
    
    def forward(self, image_features: torch.Tensor, 
                text_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Full forward pass through vision-language transformer.
        
        Args:
            image_features: (batch, image_dim)
            text_features: (batch, seq_len, text_dim)
        
        Returns:
            Dictionary with all task outputs:
            - 'vision_emb': Fused vision embeddings
            - 'text_emb': Fused text embeddings
            - 'itm_logits': Image-text matching logits
            - 'mlm_logits': Masked language modeling logits
            - 'contrastive_logits': Contrastive learning logits
        """
        # ========== Single-Modal Encoding ==========
        vision_emb = self.encode_vision(image_features)
        text_emb = self.encode_text(text_features)
        
        # ========== Cross-Modal Fusion ==========
        vision_fused, text_fused = self.cross_modal_fusion(vision_emb, text_emb)
        
        # ========== Task-Specific Outputs ==========
        
        # 1. Image-Text Matching (ITM)
        # Pool vision and text embeddings
        vision_pooled = vision_fused.mean(dim=1)  # (batch, embed_dim)
        text_pooled = text_fused.mean(dim=1)      # (batch, embed_dim)
        
        # Concatenate and classify
        multimodal_rep = torch.cat([vision_pooled, text_pooled], dim=1)
        itm_logits = self.itm_head(multimodal_rep)  # (batch, 2)
        
        # 2. Masked Language Modeling (MLM)
        # Predict each text token given multimodal context
        mlm_logits = self.mlm_head(text_fused)  # (batch, seq_len, vocab_size)
        
        # 3. Image-Text Contrastive (ITC)
        # Project to contrastive space and normalize
        vision_proj = F.normalize(self.vision_proj_head(vision_pooled), p=2, dim=1)
        text_proj = F.normalize(self.text_proj_head(text_pooled), p=2, dim=1)
        
        # Compute temperature-scaled logits
        logit_scale = torch.clamp(self.logit_scale.exp(), max=100)
        contrastive_logits = logit_scale * torch.matmul(vision_proj, text_proj.t())
        
        return {
            'vision_emb': vision_fused,
            'text_emb': text_fused,
            'itm_logits': itm_logits,
            'mlm_logits': mlm_logits,
            'contrastive_logits': contrastive_logits,
            'vision_proj': vision_proj,
            'text_proj': text_proj
        }


# ============================================================================
# PART 3: MULTI-TASK TRAINING
# ============================================================================

class MultiTaskLoss(nn.Module):
    """
    Combined loss for multi-task vision-language pretraining.
    
    Three objectives:
    1. Image-Text Contrastive (ITC): Like CLIP, maximize similarity for matched pairs
    2. Image-Text Matching (ITM): Binary classification of matched/mismatched pairs
    3. Masked Language Modeling (MLM): Predict masked words given image context
    
    Total Loss:
        L = λ_itc * L_itc + λ_itm * L_itm + λ_mlm * L_mlm
    
    Args:
        lambda_itc: Weight for contrastive loss
        lambda_itm: Weight for matching loss
        lambda_mlm: Weight for language modeling loss
    """
    
    def __init__(self, lambda_itc: float = 1.0, lambda_itm: float = 1.0,
                 lambda_mlm: float = 1.0):
        super(MultiTaskLoss, self).__init__()
        self.lambda_itc = lambda_itc
        self.lambda_itm = lambda_itm
        self.lambda_mlm = lambda_mlm
        
    def contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """
        InfoNCE contrastive loss (symmetric).
        
        Same as intermediate tutorial but applied here.
        """
        batch_size = logits.shape[0]
        labels = torch.arange(batch_size, device=logits.device)
        
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)
        
        return (loss_i2t + loss_t2i) / 2
    
    def itm_loss(self, itm_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Binary cross-entropy for image-text matching.
        
        Args:
            itm_logits: Predicted logits (batch, 2)
            labels: True labels (batch,) - 0 or 1
        """
        return F.cross_entropy(itm_logits, labels)
    
    def mlm_loss(self, mlm_logits: torch.Tensor, target_ids: torch.Tensor,
                 mask: torch.Tensor) -> torch.Tensor:
        """
        Cross-entropy for masked language modeling.
        
        Only compute loss on masked positions.
        
        Args:
            mlm_logits: Predicted logits (batch, seq_len, vocab_size)
            target_ids: True token IDs (batch, seq_len)
            mask: Boolean mask for masked positions (batch, seq_len)
        """
        # Flatten tensors for cross_entropy
        mlm_logits = mlm_logits.view(-1, mlm_logits.size(-1))
        target_ids = target_ids.view(-1)
        mask = mask.view(-1)
        
        # Compute loss only on masked positions
        loss = F.cross_entropy(mlm_logits, target_ids, reduction='none')
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        
        return loss
    
    def forward(self, outputs: Dict[str, torch.Tensor],
                itm_labels: torch.Tensor, mlm_targets: torch.Tensor,
                mlm_mask: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total multi-task loss.
        
        Args:
            outputs: Model outputs dictionary
            itm_labels: ITM labels (batch,)
            mlm_targets: MLM target token IDs (batch, seq_len)
            mlm_mask: MLM mask (batch, seq_len)
        
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Individual losses
        loss_itc = self.contrastive_loss(outputs['contrastive_logits'])
        loss_itm = self.itm_loss(outputs['itm_logits'], itm_labels)
        loss_mlm = self.mlm_loss(outputs['mlm_logits'], mlm_targets, mlm_mask)
        
        # Combined loss
        total_loss = (self.lambda_itc * loss_itc +
                     self.lambda_itm * loss_itm +
                     self.lambda_mlm * loss_mlm)
        
        # Return losses for logging
        loss_dict = {
            'total': total_loss.item(),
            'itc': loss_itc.item(),
            'itm': loss_itm.item(),
            'mlm': loss_mlm.item()
        }
        
        return total_loss, loss_dict


# ============================================================================
# PART 4: ENHANCED DATASET WITH MULTI-TASK LABELS
# ============================================================================

class MultiTaskVisionLanguageDataset(Dataset):
    """
    Dataset for multi-task vision-language pretraining.
    
    For each sample, provides:
    1. Image features
    2. Text features
    3. ITM label (matched or mismatched)
    4. MLM target tokens and mask
    
    In practice, you'd use real datasets like MS-COCO with actual images and text.
    """
    
    def __init__(self, num_samples: int = 500, image_dim: int = 2048,
                 text_dim: int = 768, seq_len: int = 20, vocab_size: int = 30000,
                 num_classes: int = 10, mask_prob: float = 0.15):
        super(MultiTaskVisionLanguageDataset, self).__init__()
        
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.mask_prob = mask_prob
        
        # Generate data
        self.image_features = []
        self.text_features = []
        self.token_ids = []
        self.labels = []
        
        for class_idx in range(num_classes):
            # Class centers
            img_center = torch.randn(image_dim) * 2
            txt_center = torch.randn(text_dim) * 2
            
            for _ in range(num_samples // num_classes):
                # Generate image
                img = img_center + torch.randn(image_dim) * 0.5
                
                # Generate text sequence
                txt_seq = txt_center.unsqueeze(0).repeat(seq_len, 1)
                txt_seq = txt_seq + torch.randn(seq_len, text_dim) * 0.3
                
                # Generate token IDs (random for toy data)
                tokens = torch.randint(0, vocab_size, (seq_len,))
                
                self.image_features.append(img)
                self.text_features.append(txt_seq)
                self.token_ids.append(tokens)
                self.labels.append(class_idx)
        
        self.image_features = torch.stack(self.image_features)
        self.text_features = torch.stack(self.text_features)
        self.token_ids = torch.stack(self.token_ids)
        self.labels = torch.tensor(self.labels)
        
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample with all task labels.
        
        Returns dictionary with:
        - image_features
        - text_features
        - itm_label (0=mismatch, 1=match)
        - mlm_tokens (original tokens)
        - mlm_mask (which positions are masked)
        """
        image = self.image_features[idx]
        text = self.text_features[idx]
        tokens = self.token_ids[idx].clone()
        
        # ITM label: 50% chance of negative pair
        if np.random.rand() < 0.5:
            # Negative pair: replace text with random sample
            random_idx = np.random.randint(0, len(self))
            text = self.text_features[random_idx]
            tokens = self.token_ids[random_idx].clone()
            itm_label = 0  # Mismatch
        else:
            itm_label = 1  # Match
        
        # MLM: randomly mask tokens
        mlm_mask = torch.rand(self.seq_len) < self.mask_prob
        mlm_targets = tokens.clone()
        
        # Replace masked positions with special [MASK] token (use vocab_size-1)
        tokens[mlm_mask] = self.vocab_size - 1
        
        return {
            'image_features': image,
            'text_features': text,
            'text_tokens': tokens,  # With masking applied
            'itm_label': itm_label,
            'mlm_targets': mlm_targets,  # Original tokens
            'mlm_mask': mlm_mask.float()  # Masked positions
        }


# ============================================================================
# PART 5: TRAINING FUNCTION
# ============================================================================

def train_vision_language_transformer(model: VisionLanguageTransformer,
                                     train_loader: DataLoader,
                                     num_epochs: int = 20,
                                     learning_rate: float = 1e-4,
                                     device: str = 'cpu') -> dict:
    """
    Train vision-language transformer with multi-task objectives.
    """
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    criterion = MultiTaskLoss(lambda_itc=1.0, lambda_itm=1.0, lambda_mlm=1.0)
    
    history = {
        'total_loss': [],
        'itc_loss': [],
        'itm_loss': [],
        'mlm_loss': []
    }
    
    print("=" * 70)
    print("TRAINING VISION-LANGUAGE TRANSFORMER")
    print("=" * 70)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = {'total': 0, 'itc': 0, 'itm': 0, 'mlm': 0}
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch in pbar:
            images = batch['image_features'].to(device)
            texts = batch['text_features'].to(device)
            itm_labels = batch['itm_label'].to(device)
            mlm_targets = batch['mlm_targets'].to(device)
            mlm_mask = batch['mlm_mask'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images, texts)
            
            # Compute multi-task loss
            loss, loss_dict = criterion(outputs, itm_labels, mlm_targets, mlm_mask)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Track losses
            for key in epoch_losses:
                epoch_losses[key] += loss_dict[key]
            num_batches += 1
            
            pbar.set_postfix({k: f'{v:.4f}' for k, v in loss_dict.items()})
        
        scheduler.step()
        
        # Epoch summary
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
            history[f'{key}_loss'].append(epoch_losses[key])
        
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        print(f"  Total Loss: {epoch_losses['total']:.4f}")
        print(f"  ITC Loss: {epoch_losses['itc']:.4f}")
        print(f"  ITM Loss: {epoch_losses['itm']:.4f}")
        print(f"  MLM Loss: {epoch_losses['mlm']:.4f}")
        print("-" * 70)
    
    return history


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function demonstrating vision-language transformer."""
    print("=" * 70)
    print("VISION-LANGUAGE TRANSFORMER - ADVANCED TUTORIAL")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Configuration
    config = {
        'image_dim': 2048,
        'text_dim': 768,
        'embed_dim': 512,
        'seq_len': 20,
        'vocab_size': 30000,
        'batch_size': 32,
        'num_epochs': 15,
        'learning_rate': 1e-4
    }
    
    # Create dataset
    print("\nCreating dataset...")
    dataset = MultiTaskVisionLanguageDataset(
        num_samples=500,
        image_dim=config['image_dim'],
        text_dim=config['text_dim'],
        seq_len=config['seq_len'],
        vocab_size=config['vocab_size']
    )
    
    train_loader = DataLoader(dataset, batch_size=config['batch_size'], 
                            shuffle=True, num_workers=0)
    
    # Initialize model
    print("\nInitializing Vision-Language Transformer...")
    model = VisionLanguageTransformer(
        image_dim=config['image_dim'],
        text_dim=config['text_dim'],
        embed_dim=config['embed_dim'],
        num_layers_single=2,
        num_layers_cross=2,
        vocab_size=config['vocab_size'],
        max_text_len=config['seq_len']
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Train model
    history = train_vision_language_transformer(
        model=model,
        train_loader=train_loader,
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate'],
        device=device
    )
    
    # Plot training curves
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].plot(history['total_loss'], marker='o')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(history['itc_loss'], marker='o', color='orange')
    axes[0, 1].set_title('Contrastive Loss (ITC)')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(history['itm_loss'], marker='o', color='green')
    axes[1, 0].set_title('Matching Loss (ITM)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(history['mlm_loss'], marker='o', color='red')
    axes[1, 1].set_title('Language Modeling Loss (MLM)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('vl_transformer_training.png', dpi=300, bbox_inches='tight')
    print("\nTraining curves saved as 'vl_transformer_training.png'")
    
    print("\n" + "=" * 70)
    print("TUTORIAL COMPLETED!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("1. Cross-modal attention enables vision-language interaction")
    print("2. Multi-task learning combines complementary objectives")
    print("3. ITM provides dense supervision for alignment")
    print("4. MLM grounds language in visual context")
    print("5. Architecture supports diverse downstream tasks")
    print("\nApplications:")
    print("- Image Captioning: Generate descriptions of images")
    print("- Visual Question Answering: Answer questions about images")
    print("- Visual Grounding: Localize objects mentioned in text")
    print("- Image-Text Retrieval: Search across modalities")
    print("\nFamous Models Using This Architecture:")
    print("- ViLBERT, LXMERT (early two-stream models)")
    print("- ALBEF, BLIP (modern unified models)")
    print("- Flamingo (few-shot visual reasoning)")


if __name__ == "__main__":
    main()
