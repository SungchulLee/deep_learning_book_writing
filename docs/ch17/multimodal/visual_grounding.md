# Visual Grounding

## Learning Objectives

By the end of this section, you will be able to:

- Understand visual grounding as localizing image regions from natural language descriptions
- Distinguish referring expression comprehension from phrase grounding
- Describe one-stage and two-stage grounding architectures

## Problem Definition

Visual grounding (also called referring expression comprehension) maps a natural language expression to a specific region in an image:

**Input**: Image + text query (e.g., "the red car on the left")
**Output**: Bounding box or segmentation mask of the referred object

This is the inverse of image captioning—instead of generating text from an image, we localize an image region from text.

## Task Variants

| Task | Input | Output |
|------|-------|--------|
| Referring Expression Comprehension | Image + expression | Single bounding box |
| Phrase Grounding | Image + sentence | Box per noun phrase |
| Referring Expression Segmentation | Image + expression | Pixel-level mask |

## Approaches

### Two-Stage

1. Generate candidate regions (proposals or grid)
2. Rank regions by similarity to the text query

### One-Stage (End-to-End)

Directly predict bounding box coordinates from image-text features, often using transformer-based architectures that jointly process visual and textual tokens.

### Modern Approaches

CLIP-based methods and grounding-specific models (Grounding DINO, GLIP) achieve strong zero-shot grounding by leveraging vision-language pre-training.

## Key Datasets

| Dataset | Images | Expressions | Avg Length |
|---------|--------|-------------|-----------|
| RefCOCO | 19,994 | 142,209 | 3.5 words |
| RefCOCO+ | 19,994 | 141,564 | 3.5 words |
| RefCOCOg | 26,711 | 104,560 | 8.4 words |

## References

1. Yu, L., et al. (2016). Modeling Context in Referring Expressions. ECCV.
2. Liu, S., et al. (2023). Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection. arXiv.
