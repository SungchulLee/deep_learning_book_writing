"""
Vision Transformer (ViT)
"""
import torch
import torch.nn as nn
from patch_embedding import PatchEmbedding

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=1000, d_model=768, 
                 num_heads=12, num_layers=12):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, embed_dim=d_model)
        
        # Class token and position embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, self.patch_embed.n_patches + 1, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, dim_feedforward=d_model*4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Classification head
        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)  # [B, n_patches, d_model]
        
        # Add class token
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embeddings
        x = x + self.pos_embed
        
        # Transformer
        x = x.transpose(0, 1)  # [seq, batch, dim]
        x = self.transformer(x)
        x = x.transpose(0, 1)  # [batch, seq, dim]
        
        # Classification
        cls_output = x[:, 0]
        return self.classifier(cls_output)
