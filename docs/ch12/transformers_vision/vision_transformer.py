"""vision_transformer — vision_transformer 페이지의 코드를 모듈로 쓸 수 있게 옮겨 놓은 것이다.

이 파일은 ch10/transformers_vision/vision_transformer.md 에서 자동으로 뽑아낸 것이므로, 고칠 일이 있으면
그 페이지를 고치고 다시 뽑아내는 편이 낫다.
"""

"""
비전 트랜스포머 (ViT)
"""
import torch
import torch.nn as nn
from patch_embedding import PatchEmbedding

# ========================================================================
# 메인
# ========================================================================

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=1000, d_model=768, 
                 num_heads=12, num_layers=12):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, embed_dim=d_model)
        
        # 분류 토큰과 자리 임베딩
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, self.patch_embed.n_patches + 1, d_model))
        
        # 트랜스포머 인코더
        encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, dim_feedforward=d_model*4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 분류 머리
        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        # 조각 임베딩
        x = self.patch_embed(x)  # [B, n_patches, d_model]
        
        # 분류 토큰을 더한다
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # 자리 임베딩을 더한다
        x = x + self.pos_embed
        
        # 트랜스포머
        x = x.transpose(0, 1)  # [seq, batch, dim]
        x = self.transformer(x)
        x = x.transpose(0, 1)  # [batch, seq, dim]
        
        # 분류
        cls_output = x[:, 0]
        return self.classifier(cls_output)


if __name__ == "__main__":
    pass
