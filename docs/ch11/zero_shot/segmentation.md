# Zero-Shot Segmentation

Zero-shot segmentation extends zero-shot learning to dense prediction tasks, enabling pixel-level classification of categories not seen during training. Modern approaches leverage vision-language models to achieve open-vocabulary segmentation.

## Open-Vocabulary Segmentation

Vision-language models like CLIP enable segmentation of arbitrary categories by matching pixel-level features against text descriptions:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ZeroShotSegmentor(nn.Module):
    """
    Zero-shot semantic segmentation using vision-language alignment.
    
    Matches dense visual features against text embeddings of class names.
    """
    
    def __init__(self, visual_encoder, text_encoder, feature_dim=512):
        super().__init__()
        self.visual_encoder = visual_encoder
        self.text_encoder = text_encoder
        self.projection = nn.Conv2d(feature_dim, feature_dim, 1)
    
    def forward(self, images, class_names):
        """
        Args:
            images: (B, 3, H, W) input images
            class_names: List of class name strings
        
        Returns:
            segmentation_logits: (B, num_classes, H, W)
        """
        # Extract dense visual features
        visual_features = self.visual_encoder(images)  # (B, D, h, w)
        visual_features = self.projection(visual_features)
        visual_features = F.normalize(visual_features, dim=1)
        
        # Encode class names
        text_embeddings = self.text_encoder(class_names)  # (num_classes, D)
        text_embeddings = F.normalize(text_embeddings, dim=1)
        
        # Compute similarity maps
        B, D, h, w = visual_features.shape
        visual_flat = visual_features.view(B, D, -1)  # (B, D, h*w)
        
        # (B, num_classes, h*w)
        similarity = torch.einsum('bdn,cd->bcn', visual_flat, text_embeddings)
        
        # Reshape to spatial
        logits = similarity.view(B, len(class_names), h, w)
        
        # Upsample to input resolution
        logits = F.interpolate(logits, size=images.shape[-2:], mode='bilinear')
        
        return logits
```

## Key Methods

| Method | Approach | Backbone |
|--------|----------|----------|
| LSeg | CLIP + pixel-text alignment | ViT / ResNet |
| GroupViT | Learned grouping without pixel labels | ViT |
| OpenSeg | Region-word grounding | EfficientNet |
| SAM + CLIP | Segment Anything + CLIP classification | ViT-H |
| FC-CLIP | Frozen CLIP for open-vocabulary segmentation | CLIP ViT |

## Applications

### Rare Disease Diagnosis

```python
class MedicalZSLClassifier:
    """
    Zero-shot classifier for rare medical conditions.
    """
    
    def __init__(self, model, symptom_embeddings, disease_embeddings):
        self.model = model
        self.symptom_emb = symptom_embeddings
        self.disease_emb = disease_embeddings
    
    def diagnose(self, patient_features, known_diseases, rare_diseases):
        """
        Diagnose patient considering both common and rare diseases.
        
        Args:
            patient_features: Clinical/imaging features
            known_diseases: Diseases with training data
            rare_diseases: Rare diseases without training data
        
        Returns:
            Top-k diagnoses with confidence scores
        """
        self.model.eval()
        
        with torch.no_grad():
            # Get all disease embeddings
            all_diseases = known_diseases + rare_diseases
            all_emb = torch.stack([
                torch.tensor(self.disease_emb[d], dtype=torch.float32)
                for d in all_diseases
            ])
            
            # Compute compatibility scores
            scores = self.model(patient_features, all_emb)
            probs = F.softmax(scores, dim=-1)
            
            # Get top predictions
            top_probs, top_indices = probs.topk(5)
            
            diagnoses = [
                {'disease': all_diseases[i], 
                 'confidence': p.item(),
                 'is_rare': all_diseases[i] in rare_diseases}
                for i, p in zip(top_indices[0], top_probs[0])
            ]
            
            return diagnoses
```

### Fine-Grained Product Classification

```python
class ProductZSLClassifier:
    """
    Zero-shot classifier for e-commerce product categorization.
    """
    
    def __init__(self, clip_model):
        self.model = clip_model
    
    def classify_product(self, image, category_hierarchy):
        """
        Classify product into fine-grained categories.
        
        Args:
            image: Product image
            category_hierarchy: Dict of {parent: [child_categories]}
        
        Returns:
            Hierarchical classification
        """
        results = {}
        
        # Classify at each level
        current_candidates = list(category_hierarchy.keys())
        
        while current_candidates:
            # Create prompts
            prompts = [f"a photo of {cat}" for cat in current_candidates]
            
            # CLIP classification
            pred, probs = clip_zero_shot_classify(image, current_candidates)
            
            results[pred] = probs.max()
            
            # Go to children if they exist
            if pred in category_hierarchy:
                current_candidates = category_hierarchy[pred]
            else:
                break
        
        return results
```

### Wildlife Species Recognition

```python
def wildlife_zsl_system(image_path, known_species, rare_species, 
                        attribute_embeddings, clip_model):
    """
    Zero-shot wildlife species recognition.
    
    Combines visual features with semantic attributes for
    recognizing rare species.
    """
    # Extract CLIP features
    image_features = extract_clip_features(clip_model, image_path)
    
    # Get attribute-based predictions for rare species
    rare_scores = []
    for species in rare_species:
        # Use species description/attributes
        description = f"a photo of a {species}, which is a wild animal"
        text_features = encode_text(clip_model, description)
        
        score = cosine_similarity(image_features, text_features)
        rare_scores.append(score)
    
    # Combine with standard classification for known species
    known_probs = clip_classify(clip_model, image_path, known_species)
    
    # Calibrate scores
    calibration = find_optimal_calibration(known_probs, rare_scores)
    
    # Return combined predictions
    all_species = known_species + rare_species
    all_scores = np.concatenate([known_probs, np.array(rare_scores) + calibration])
    
    return all_species[all_scores.argmax()], all_scores
```

## References

1. Li, B., et al. (2022). "Language-Driven Semantic Segmentation." *ICLR*.
2. Xu, J., et al. (2022). "GroupViT: Semantic Segmentation Emerges from Text Supervision." *CVPR*.
3. Kirillov, A., et al. (2023). "Segment Anything." *ICCV*.
