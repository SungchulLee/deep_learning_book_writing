# Zero-Shot Learning with CLIP

## CLIP Overview

CLIP (Contrastive Language-Image Pre-training) by OpenAI revolutionized zero-shot learning by training on 400 million image-text pairs:

$$\mathcal{L}_{CLIP} = -\frac{1}{N} \sum_{i=1}^{N} \left[ \log \frac{\exp(\text{sim}(\mathbf{v}_i, \mathbf{t}_i) / \tau)}{\sum_j \exp(\text{sim}(\mathbf{v}_i, \mathbf{t}_j) / \tau)} + \log \frac{\exp(\text{sim}(\mathbf{t}_i, \mathbf{v}_i) / \tau)}{\sum_j \exp(\text{sim}(\mathbf{t}_i, \mathbf{v}_j) / \tau)} \right]$$

## Zero-Shot Classification with CLIP

```python
import torch
from PIL import Image
import clip

def clip_zero_shot_classify(image_path, class_names, prompt_template="a photo of a {}"):
    """
    Zero-shot classification using CLIP.
    
    Args:
        image_path: Path to image file
        class_names: List of candidate class names
        prompt_template: Template for text prompts
    
    Returns:
        Predicted class and confidence scores
    """
    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    # Preprocess image
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    
    # Create text prompts
    text_prompts = [prompt_template.format(name) for name in class_names]
    text_tokens = clip.tokenize(text_prompts).to(device)
    
    # Encode image and text
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_tokens)
        
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Compute similarity
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    
    # Get prediction
    probs = similarity[0].cpu().numpy()
    pred_idx = probs.argmax()
    
    return class_names[pred_idx], probs


def clip_with_ensemble_prompts(image_path, class_names):
    """
    CLIP with multiple prompt templates for improved accuracy.
    """
    templates = [
        "a photo of a {}",
        "a blurry photo of a {}",
        "a cropped photo of a {}",
        "a bright photo of a {}",
        "a dark photo of a {}",
        "a close-up photo of a {}",
        "a photo of the {}",
        "a good photo of a {}",
        "a rendition of a {}",
        "a photo of the large {}",
        "a photo of the small {}"
    ]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    
    # Average text features across templates
    all_text_features = []
    
    for template in templates:
        text_prompts = [template.format(name) for name in class_names]
        text_tokens = clip.tokenize(text_prompts).to(device)
        
        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            all_text_features.append(text_features)
    
    # Average across templates
    text_features = torch.stack(all_text_features).mean(dim=0)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # Compute similarity
    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    
    probs = similarity[0].cpu().numpy()
    return class_names[probs.argmax()], probs
```

## Fine-Tuning CLIP for ZSL

```python
class CLIPFineTuner(nn.Module):
    """
    Fine-tune CLIP for domain-specific ZSL.
    """
    
    def __init__(self, clip_model, n_ctx=4, class_names=None):
        super().__init__()
        
        self.clip = clip_model
        self.n_ctx = n_ctx
        
        # Learnable context vectors (prompt tuning)
        ctx_dim = self.clip.ln_final.weight.shape[0]
        ctx_vectors = torch.empty(n_ctx, ctx_dim)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)
        
        # Store class name embeddings
        if class_names is not None:
            self.register_class_names(class_names)
    
    def register_class_names(self, class_names):
        """Pre-compute class name token embeddings."""
        with torch.no_grad():
            prompts = [f"a photo of a {name}" for name in class_names]
            text_tokens = clip.tokenize(prompts)
            
            # Get token embeddings (before transformer)
            self.name_embeddings = self.clip.token_embedding(text_tokens)
    
    def forward(self, images):
        """
        Forward pass with learned prompts.
        """
        # Encode images
        image_features = self.clip.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Build text features with learnable context
        batch_size = self.name_embeddings.shape[0]
        
        # Insert context before class name
        prefix = self.ctx.unsqueeze(0).expand(batch_size, -1, -1)
        prompts = torch.cat([
            self.name_embeddings[:, :1],  # SOS token
            prefix,
            self.name_embeddings[:, 1:]   # Class name + EOS
        ], dim=1)
        
        # Encode text through transformer
        text_features = self.clip_text_encoder(prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Compute logits
        logits = 100.0 * image_features @ text_features.T
        
        return logits
```

## References

1. Radford, A., et al. (2021). "Learning Transferable Visual Models From Natural Language Supervision." *ICML*.
2. Jia, C., et al. (2021). "Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision." *ICML*.
