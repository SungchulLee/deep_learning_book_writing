# Module 35: Multimodal Vision

## Overview
This module explores multimodal learning combining vision and language, focusing on models that can understand and relate images with text. Topics include contrastive learning, vision-language pretraining, cross-modal attention, and applications like zero-shot classification and image-text retrieval.

## Mathematical Foundations

### 1. Multimodal Embedding Space
The goal is to learn a joint embedding space where related images and text are close:

**Image Encoder**: f_v: I → R^d
**Text Encoder**: f_t: T → R^d

where I is the image space, T is the text space, and d is the embedding dimension.

### 2. Contrastive Learning (CLIP-style)
Given N (image, text) pairs, maximize similarity for matched pairs and minimize for mismatched pairs.

**Similarity**: s(i,j) = (f_v(I_i)^T · f_t(T_j)) / (||f_v(I_i)|| · ||f_t(T_j)||)

**InfoNCE Loss** (for images):
L_i = -log(exp(s(i,i)/τ) / Σ_j exp(s(i,j)/τ))

**Symmetric Loss**: L = (L_image + L_text) / 2

where τ is the temperature parameter.

### 3. Cross-Modal Attention
Attention weights for attending to image features given text query:

**Attention Scores**: A = softmax((Q·K^T) / √d_k)

where:
- Q = W_q·f_t(text) (query from text)
- K = W_k·f_v(image) (keys from image)
- V = W_v·f_v(image) (values from image)

**Output**: O = A·V

### 4. Vision-Language Pretraining Objectives

**Masked Language Modeling (MLM)**:
Predict masked tokens in text conditioned on image: P(t_i | t_{\i}, I)

**Image-Text Matching (ITM)**:
Binary classification whether (image, text) is a matched pair: P(match | I, T)

**Masked Region Modeling (MRM)**:
Predict masked image regions conditioned on text: P(r_i | r_{\i}, T)

### 5. Zero-Shot Classification
For K classes with text descriptions {T_1, ..., T_K}:

**Class Probability**: P(y=k|I) = exp(s(I,T_k)/τ) / Σ_j exp(s(I,T_j)/τ)

## Module Structure

### Beginner Level: `01_multimodal_embeddings_beginner.py`
- Simple dual-encoder architecture
- Basic image-text pairing
- Cosine similarity matching
- Introduction to contrastive loss
- **Concepts**: Joint embeddings, similarity metrics, basic retrieval

### Intermediate Level: `02_clip_style_contrastive_intermediate.py`
- CLIP-style contrastive learning
- Batch-wise contrastive loss (InfoNCE)
- Temperature-scaled similarities
- Zero-shot image classification
- Image-text retrieval in both directions
- **Concepts**: Contrastive learning, symmetric loss, temperature scaling

### Advanced Level: `03_vision_language_transformer_advanced.py`
- Full vision-language transformer
- Cross-modal attention mechanisms
- Multi-task pretraining (ITM + MLM)
- Vision-language fusion strategies
- Image captioning and VQA
- **Concepts**: Cross-attention, multimodal fusion, transformer architectures

### Utilities: `multimodal_utils.py`
- Data loading for vision-language pairs
- Text tokenization and processing
- Image preprocessing pipelines
- Evaluation metrics (recall@K, mAP)
- Visualization tools

## Key Concepts Covered

### 1. **Dual-Encoder Architecture**
   - Separate encoders for vision and language
   - Shared embedding space
   - Efficient for large-scale retrieval

### 2. **Contrastive Learning**
   - Positive and negative pairs
   - InfoNCE loss formulation
   - Temperature scaling effects
   - Batch construction strategies

### 3. **Cross-Modal Attention**
   - Query-key-value mechanism across modalities
   - Early vs. late fusion
   - Attention visualization

### 4. **Zero-Shot Transfer**
   - Text-based class descriptions
   - No task-specific fine-tuning
   - Prompt engineering for vision tasks

### 5. **Multimodal Pretraining**
   - Large-scale vision-language datasets
   - Self-supervised objectives
   - Transfer to downstream tasks

## Applications

1. **Image-Text Retrieval**: Find images matching text queries or vice versa
2. **Zero-Shot Classification**: Classify images without training examples
3. **Visual Question Answering**: Answer questions about image content
4. **Image Captioning**: Generate natural language descriptions
5. **Visual Grounding**: Localize objects mentioned in text
6. **Multimodal Search**: Search across vision and language

## Famous Models & Papers

### Foundational Work:
- **CLIP** (Radford et al., 2021): Contrastive Language-Image Pretraining
- **ALIGN** (Jia et al., 2021): Large-scale noisy image-text pairs
- **FLAVA** (Singh et al., 2022): Unified vision-language model

### Advanced Architectures:
- **ViLT** (Kim et al., 2021): Vision-and-Language Transformer
- **ALBEF** (Li et al., 2021): Align before Fuse
- **BLIP** (Li et al., 2022): Bootstrapping Language-Image Pretraining
- **CoCa** (Yu et al., 2022): Contrastive Captioners

### Applications:
- **Flamingo** (Alayrac et al., 2022): Few-shot visual question answering
- **DALL-E** (Ramesh et al., 2021): Text-to-image generation
- **GLIDE** (Nichol et al., 2022): Guided diffusion for image generation

## Prerequisites
- Module 02: Tensors and basic operations
- Module 20: Feedforward networks
- Module 23: Convolutional neural networks
- Module 25: Attention mechanisms
- Module 26: Transformers for NLP
- Module 27: Vision Transformers

## Learning Objectives

By completing this module, students will:
1. Understand joint embedding spaces for vision and language
2. Implement contrastive learning for multimodal data
3. Build dual-encoder architectures (CLIP-style)
4. Design cross-modal attention mechanisms
5. Apply zero-shot transfer to vision tasks
6. Evaluate multimodal retrieval systems
7. Understand vision-language pretraining objectives

## Datasets for Practice

### Image-Text Datasets:
- **MS-COCO**: 330K images with 5 captions each
- **Flickr30K**: 31K images with 5 captions each
- **Conceptual Captions**: 3.3M image-caption pairs (noisy)
- **LAION-400M**: 400M image-text pairs from web

### Task-Specific:
- **VQA v2**: Visual question answering (1M questions)
- **Visual Genome**: Dense annotations (5.4M region descriptions)
- **RefCOCO**: Referring expression comprehension

## Installation Requirements
```bash
pip install torch torchvision transformers pillow matplotlib numpy tqdm scikit-learn
```

## Usage Examples

### Basic Image-Text Matching:
```python
from beginner import SimpleMultimodalModel
model = SimpleMultimodalModel(image_dim=2048, text_dim=768, embed_dim=512)
image_features = model.image_encoder(images)
text_features = model.text_encoder(texts)
similarity = model.compute_similarity(image_features, text_features)
```

### CLIP-style Training:
```python
from intermediate import CLIPModel, ContrastiveLoss
model = CLIPModel(embed_dim=512, temperature=0.07)
loss_fn = ContrastiveLoss(temperature=0.07)
loss = loss_fn(image_features, text_features)
```

### Zero-Shot Classification:
```python
# Define class descriptions
class_texts = ["a photo of a dog", "a photo of a cat", ...]
text_features = model.encode_text(class_texts)
image_features = model.encode_image(image)
probs = model.zero_shot_classifier(image_features, text_features)
```

## Performance Metrics

### Retrieval Metrics:
- **Recall@K**: Fraction of queries with correct result in top-K
- **Mean Reciprocal Rank (MRR)**: Average of 1/rank for correct results
- **Mean Average Precision (mAP)**: Average precision across all queries

### Classification Metrics:
- **Zero-shot Accuracy**: Classification without training examples
- **Top-5 Accuracy**: Correct class in top 5 predictions

## Training Tips

1. **Large Batch Sizes**: Contrastive learning benefits from many negatives (256-1024)
2. **Temperature Tuning**: Start with τ=0.07, adjust based on embedding scale
3. **Symmetric Loss**: Use both image→text and text→image directions
4. **Data Augmentation**: Random crops, color jitter for images; back-translation for text
5. **Warmup**: Use learning rate warmup for stable training
6. **Gradient Clipping**: Prevent exploding gradients in transformer models

## Challenges & Research Directions

1. **Compositionality**: Understanding complex scene descriptions
2. **Fine-grained Understanding**: Distinguishing subtle visual differences
3. **Grounding**: Connecting language to specific image regions
4. **Efficiency**: Scaling to billions of image-text pairs
5. **Multilingual**: Extending to non-English languages
6. **Video-Language**: Temporal understanding in videos

## Additional Resources

### Tutorials:
- [OpenAI CLIP Repository](https://github.com/openai/CLIP)
- [Hugging Face Vision-Language Models](https://huggingface.co/models?pipeline_tag=image-to-text)

### Papers:
- "Learning Transferable Visual Models From Natural Language Supervision" (CLIP)
- "Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision" (ALIGN)
- "BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding"

### Courses:
- Stanford CS231n: Convolutional Neural Networks for Visual Recognition
- UMich EECS 498: Deep Learning for Computer Vision (multimodal section)

## Time Estimate
- **Beginner**: 2-3 hours
- **Intermediate**: 4-5 hours  
- **Advanced**: 6-8 hours
- **Total**: 12-16 hours for complete module

## Assessment Ideas

1. **Project 1**: Build an image search engine using text queries
2. **Project 2**: Implement zero-shot object detection using text prompts
3. **Project 3**: Create a visual question answering system
4. **Project 4**: Fine-tune CLIP for domain-specific classification
5. **Project 5**: Compare different fusion strategies (early vs. late)
