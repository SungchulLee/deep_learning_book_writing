# Chapter 17: Computer Vision


!!! warning "Incomplete page"
    This page is missing the required five-section structure (Concept Definition, Explanation, Diagram / Example). Content needs to be reorganized and expanded.

This chapter covers the major deep learning tasks and architectures in computer vision, from image classification through object detection, semantic segmentation, video understanding, and multimodal vision-language models. Each section traces the evolution of key architectures, presents the underlying mathematical formulations, and provides practical PyTorch implementations.

---

## Image Classification

Foundational architectures for assigning labels to images, from early convolutional networks to modern efficient designs.

- Classification Overview -- The image classification task, pipeline, and architecture evolution
- LeNet -- The first successful CNN for digit recognition (1998)
- AlexNet -- Deep CNN with GPU training, ReLU, and dropout (2012)
- [VGGNet](image_classification/vgg.md) -- Uniform 3x3 convolutions and very deep networks (2014)
- GoogLeNet/Inception -- Inception modules and multi-scale processing (2014)
- ResNet -- Residual connections enabling 100+ layer networks (2015)
- DenseNet -- Dense connections for maximum feature reuse (2017)
- MobileNet -- Depthwise separable convolutions for mobile deployment (2017)
- ResNeXt -- Grouped convolutions with cardinality dimension (2017)
- ShuffleNet -- Channel shuffle for cross-group information flow (2018)
- EfficientNet -- Compound scaling of depth, width, and resolution (2019)
- NFNet -- Normalizer-free networks removing batch normalization (2021)
- ConvNeXt -- Modernized ConvNet matching Vision Transformer performance (2022)
- Fine-Grained Classification -- Distinguishing visually similar sub-categories
- Data Augmentation for Images -- Geometric, photometric, and learned augmentation strategies
- Well-Known Models Overview -- Collection of 50 influential neural network architectures
- Advanced Image Classification -- Module overview for advanced CNN architectures

---

## Segmentation

Pixel-level prediction architectures for semantic, instance, and panoptic segmentation.

- [Segmentation Fundamentals](segmentation/segmentation_overview.md) -- Pixel-wise classification, encoder-decoder paradigm, and evaluation metrics
- Semantic Segmentation -- Complete pipeline from data augmentation to production deployment
- Instance Segmentation -- Mask R-CNN and one-stage instance segmentation approaches
- Panoptic Segmentation -- Unified segmentation of "stuff" and "things"
- FCN -- Fully Convolutional Networks for dense prediction
- [U-Net](segmentation/unet.md) -- Symmetric encoder-decoder with skip connections for biomedical imaging
- DeepLab -- Atrous convolution and ASPP for multi-scale segmentation
- [Mask R-CNN](segmentation/mask_rcnn.md) -- Instance segmentation via mask prediction branch on Faster R-CNN
- Loss Functions -- Dice, Tversky, focal, and boundary-aware losses for segmentation
- Metrics -- IoU, Dice coefficient, pixel accuracy, and Panoptic Quality
- Semantic Segmentation Tutorial -- Progressive tutorial series for semantic segmentation in PyTorch
- Example 1: Basic U-Net -- Binary segmentation with U-Net from scratch
- Example 2: Pretrained Encoders -- Transfer learning with ResNet/EfficientNet backbones and DeepLabV3
- Example 3: Medical Segmentation -- Domain-specific techniques for medical imaging
- Example 4: Advanced Techniques -- Attention mechanisms, TTA, and boundary refinement

---

## Detection

Object detection architectures from two-stage region-based methods to one-stage and transformer-based detectors.

- Detection Overview -- Object detection task definition, one-stage vs two-stage paradigms
- Bounding Box Representations -- Box formats (xyxy, xywh, cxcywh) and coordinate conversions
- R-CNN -- The first deep learning-based detector with region proposals
- Fast R-CNN -- Shared computation with RoI Pooling for efficient detection
- Faster R-CNN -- Region Proposal Network for end-to-end training
- Region Proposal Networks -- Anchor generation, assignment, and proposal filtering
- [IoU and NMS](detection/iou_nms.md) -- Intersection over Union, Non-Maximum Suppression, and their variants
- [YOLO](detection/yolo.md) -- Single-pass grid-based detection from YOLOv1 through YOLOv8
- [SSD](detection/ssd.md) -- Single Shot MultiBox Detector with multi-scale feature maps
- [Focal Loss](detection/focal_loss.md) -- Addressing class imbalance in dense detection
- RetinaNet -- FPN with focal loss achieving two-stage accuracy
- [FCOS](detection/fcos.md) -- Fully convolutional anchor-free one-stage detection
- CenterNet -- Keypoint-based anchor-free, NMS-free detection
- DETR -- Detection Transformer with set prediction and Hungarian matching
- Detection Metrics -- Precision, recall, AP, mAP, and COCO evaluation protocol
- Object Detection Tutorial -- Progressive tutorial series for object detection in PyTorch
- Example 1: Basic Detection -- IoU, NMS, and anchor box fundamentals from scratch
- Example 2: YOLO Detection -- Using pretrained YOLOv8 for real-time detection
- Example 3: Custom Detection -- Training YOLO on custom objects with full pipeline
- Example 4: Advanced Techniques -- Multi-scale testing, quantization, pruning, and deployment

---

## Video

Deep learning architectures for temporal modeling, action recognition, and video understanding.

- [Video Basics](video/video_overview.md) -- Video data representation, loading, and frame sampling
- 3D Convolutions -- Spatiotemporal convolutions and 3D CNN architectures (C3D, R3D)
- [Two-Stream Networks](video/two_stream.md) -- Dual-pathway architectures for appearance and motion
- I3D -- Inflated 3D ConvNets leveraging ImageNet pre-training
- [SlowFast Networks](video/slowfast.md) -- Dual-pathway design with different temporal resolutions
- Temporal Modeling -- Temporal pooling, aggregation, and long-term dependencies
- Action Recognition -- Complete pipelines for video classification
- Video Transformers -- Space-time self-attention (TimeSformer, ViViT)
- Temporal Action Detection -- Localizing action instances in untrimmed videos
- Video Captioning -- Generating natural language descriptions from video
- Video Understanding Module -- Module overview for video understanding techniques
- Usage Guide -- Quick start and installation guide for video examples

---

## Multimodal

Vision-language models that bridge images and text for joint understanding and generation.

- Vision-Language Models -- Dual-encoder architectures, cross-modal attention, and pretraining
- CLIP -- Contrastive Language-Image Pre-training for zero-shot transfer
- ALIGN -- Large-scale noisy image-text embedding at 1.8B pair scale
- BLIP -- Bootstrapping Language-Image Pre-training with captioner-filter
- Image Captioning -- Encoder-decoder caption generation with attention
- Visual Question Answering -- Answering natural language questions about images
- Multimodal Fusion -- Early, late, and mid-level fusion strategies
- Visual Grounding -- Localizing image regions from natural language descriptions
- Cross-Modal Retrieval -- Image-to-text and text-to-image retrieval with shared embeddings
- Multimodal Vision Module -- Module overview for multimodal vision-language learning
