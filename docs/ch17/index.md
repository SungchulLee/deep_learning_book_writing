<<<<<<< HEAD
# Chapter Overview

This chapter covers **Advanced Graphs**.

# Reference

[Introduction to Algorithms (CLRS), Chapters 22, 26](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
=======
# Chapter 17: Computer Vision

This chapter covers the major deep learning tasks and architectures in computer vision, from image classification through object detection, semantic segmentation, video understanding, and multimodal vision-language models. Each section traces the evolution of key architectures, presents the underlying mathematical formulations, and provides practical PyTorch implementations.

---

## Image Classification

Foundational architectures for assigning labels to images, from early convolutional networks to modern efficient designs.

- [Classification Overview](image_classification/classification_overview.md) -- The image classification task, pipeline, and architecture evolution
- [LeNet](image_classification/lenet.md) -- The first successful CNN for digit recognition (1998)
- [AlexNet](image_classification/alexnet.md) -- Deep CNN with GPU training, ReLU, and dropout (2012)
- [VGGNet](image_classification/vgg.md) -- Uniform 3x3 convolutions and very deep networks (2014)
- [GoogLeNet/Inception](image_classification/inception.md) -- Inception modules and multi-scale processing (2014)
- [ResNet](image_classification/resnet.md) -- Residual connections enabling 100+ layer networks (2015)
- [DenseNet](image_classification/densenet.md) -- Dense connections for maximum feature reuse (2017)
- [MobileNet](image_classification/mobilenet.md) -- Depthwise separable convolutions for mobile deployment (2017)
- [ResNeXt](image_classification/resnext.md) -- Grouped convolutions with cardinality dimension (2017)
- [ShuffleNet](image_classification/shufflenet.md) -- Channel shuffle for cross-group information flow (2018)
- [EfficientNet](image_classification/efficientnet.md) -- Compound scaling of depth, width, and resolution (2019)
- [NFNet](image_classification/nfnet.md) -- Normalizer-free networks removing batch normalization (2021)
- [ConvNeXt](image_classification/convnext.md) -- Modernized ConvNet matching Vision Transformer performance (2022)
- [Fine-Grained Classification](image_classification/fine_grained.md) -- Distinguishing visually similar sub-categories
- [Data Augmentation for Images](image_classification/augmentation.md) -- Geometric, photometric, and learned augmentation strategies
- [Well-Known Models Overview](image_classification/well_known_models_overview.md) -- Collection of 50 influential neural network architectures
- [Advanced Image Classification](image_classification/image_classification_advanced_overview.md) -- Module overview for advanced CNN architectures

---

## Segmentation

Pixel-level prediction architectures for semantic, instance, and panoptic segmentation.

- [Segmentation Fundamentals](segmentation/segmentation_overview.md) -- Pixel-wise classification, encoder-decoder paradigm, and evaluation metrics
- [Semantic Segmentation](segmentation/semantic.md) -- Complete pipeline from data augmentation to production deployment
- [Instance Segmentation](segmentation/instance.md) -- Mask R-CNN and one-stage instance segmentation approaches
- [Panoptic Segmentation](segmentation/panoptic.md) -- Unified segmentation of "stuff" and "things"
- [FCN](segmentation/fcn.md) -- Fully Convolutional Networks for dense prediction
- [U-Net](segmentation/unet.md) -- Symmetric encoder-decoder with skip connections for biomedical imaging
- [DeepLab](segmentation/deeplab.md) -- Atrous convolution and ASPP for multi-scale segmentation
- [Mask R-CNN](segmentation/mask_rcnn.md) -- Instance segmentation via mask prediction branch on Faster R-CNN
- [Loss Functions](segmentation/loss_functions.md) -- Dice, Tversky, focal, and boundary-aware losses for segmentation
- [Metrics](segmentation/metrics.md) -- IoU, Dice coefficient, pixel accuracy, and Panoptic Quality
- [Semantic Segmentation Tutorial](segmentation/semantic_segmentation_overview.md) -- Progressive tutorial series for semantic segmentation in PyTorch
- [Example 1: Basic U-Net](segmentation/example_1_basic_unet_overview.md) -- Binary segmentation with U-Net from scratch
- [Example 2: Pretrained Encoders](segmentation/example_2_pretrained_encoders_overview.md) -- Transfer learning with ResNet/EfficientNet backbones and DeepLabV3
- [Example 3: Medical Segmentation](segmentation/example_3_medical_segmentation_overview.md) -- Domain-specific techniques for medical imaging
- [Example 4: Advanced Techniques](segmentation/example_4_advanced_techniques_overview.md) -- Attention mechanisms, TTA, and boundary refinement

---

## Detection

Object detection architectures from two-stage region-based methods to one-stage and transformer-based detectors.

- [Detection Overview](detection/detection_overview.md) -- Object detection task definition, one-stage vs two-stage paradigms
- [Bounding Box Representations](detection/bounding_boxes.md) -- Box formats (xyxy, xywh, cxcywh) and coordinate conversions
- [R-CNN](detection/rcnn.md) -- The first deep learning-based detector with region proposals
- [Fast R-CNN](detection/fast_rcnn.md) -- Shared computation with RoI Pooling for efficient detection
- [Faster R-CNN](detection/faster_rcnn.md) -- Region Proposal Network for end-to-end training
- [Region Proposal Networks](detection/rpn.md) -- Anchor generation, assignment, and proposal filtering
- [IoU and NMS](detection/iou_nms.md) -- Intersection over Union, Non-Maximum Suppression, and their variants
- [YOLO](detection/yolo.md) -- Single-pass grid-based detection from YOLOv1 through YOLOv8
- [SSD](detection/ssd.md) -- Single Shot MultiBox Detector with multi-scale feature maps
- [Focal Loss](detection/focal_loss.md) -- Addressing class imbalance in dense detection
- [RetinaNet](detection/retinanet.md) -- FPN with focal loss achieving two-stage accuracy
- [FCOS](detection/fcos.md) -- Fully convolutional anchor-free one-stage detection
- [CenterNet](detection/centernet.md) -- Keypoint-based anchor-free, NMS-free detection
- [DETR](detection/detr.md) -- Detection Transformer with set prediction and Hungarian matching
- [Detection Metrics](detection/metrics.md) -- Precision, recall, AP, mAP, and COCO evaluation protocol
- [Object Detection Tutorial](detection/object_detection_overview.md) -- Progressive tutorial series for object detection in PyTorch
- [Example 1: Basic Detection](detection/example_1_basic_detection_overview.md) -- IoU, NMS, and anchor box fundamentals from scratch
- [Example 2: YOLO Detection](detection/example_2_yolo_detection_overview.md) -- Using pretrained YOLOv8 for real-time detection
- [Example 3: Custom Detection](detection/example_3_custom_detection_overview.md) -- Training YOLO on custom objects with full pipeline
- [Example 4: Advanced Techniques](detection/example_4_advanced_techniques_overview.md) -- Multi-scale testing, quantization, pruning, and deployment

---

## Video

Deep learning architectures for temporal modeling, action recognition, and video understanding.

- [Video Basics](video/video_overview.md) -- Video data representation, loading, and frame sampling
- [3D Convolutions](video/conv3d.md) -- Spatiotemporal convolutions and 3D CNN architectures (C3D, R3D)
- [Two-Stream Networks](video/two_stream.md) -- Dual-pathway architectures for appearance and motion
- [I3D](video/i3d.md) -- Inflated 3D ConvNets leveraging ImageNet pre-training
- [SlowFast Networks](video/slowfast.md) -- Dual-pathway design with different temporal resolutions
- [Temporal Modeling](video/temporal_modeling.md) -- Temporal pooling, aggregation, and long-term dependencies
- [Action Recognition](video/action_recognition.md) -- Complete pipelines for video classification
- [Video Transformers](video/video_transformers.md) -- Space-time self-attention (TimeSformer, ViViT)
- [Temporal Action Detection](video/temporal_detection.md) -- Localizing action instances in untrimmed videos
- [Video Captioning](video/video_captioning.md) -- Generating natural language descriptions from video
- [Video Understanding Module](video/video_understanding_overview.md) -- Module overview for video understanding techniques
- [Usage Guide](video/usage_guide.md) -- Quick start and installation guide for video examples

---

## Multimodal

Vision-language models that bridge images and text for joint understanding and generation.

- [Vision-Language Models](multimodal/multimodal_overview.md) -- Dual-encoder architectures, cross-modal attention, and pretraining
- [CLIP](multimodal/clip.md) -- Contrastive Language-Image Pre-training for zero-shot transfer
- [ALIGN](multimodal/align.md) -- Large-scale noisy image-text embedding at 1.8B pair scale
- [BLIP](multimodal/blip.md) -- Bootstrapping Language-Image Pre-training with captioner-filter
- [Image Captioning](multimodal/image_captioning.md) -- Encoder-decoder caption generation with attention
- [Visual Question Answering](multimodal/vqa.md) -- Answering natural language questions about images
- [Multimodal Fusion](multimodal/fusion.md) -- Early, late, and mid-level fusion strategies
- [Visual Grounding](multimodal/visual_grounding.md) -- Localizing image regions from natural language descriptions
- [Cross-Modal Retrieval](multimodal/cross_modal_retrieval.md) -- Image-to-text and text-to-image retrieval with shared embeddings
- [Multimodal Vision Module](multimodal/multimodal_vision_overview.md) -- Module overview for multimodal vision-language learning
>>>>>>> 96f31bd (...)
