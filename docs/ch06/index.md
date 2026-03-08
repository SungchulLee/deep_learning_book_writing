<<<<<<< HEAD
# Chapter Overview

This chapter covers **Hashing**.

# Reference

[Introduction to Algorithms (CLRS), Chapter 11](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
=======
# Chapter 6: Convolutional Neural Networks

This chapter covers the foundational architectures for processing spatially structured data, primarily images but also time series and other grid-like inputs. We progress from classical convolutional neural networks through residual architectures to modern Vision Transformers, tracing the evolution from hand-crafted inductive biases toward learned representations.

---

## 6.1 Convolutional Neural Networks

Core CNN concepts from the convolution operation through specialized variants for efficiency and expanded receptive fields.

- [CNN Overview](cnn/cnn_overview.md) -- High-level introduction to CNNs, their structural priors, and core concepts at a glance
- [Convolutional Neural Networks Overview](cnn/convolutional_neural_networks_overview.md) -- PyTorch CNN tutorial package with progressively challenging examples
- [Convolution Operation](cnn/convolution.md) -- Mathematical foundations of discrete convolution and cross-correlation for feature extraction
- [Feature Maps](cnn/feature_maps.md) -- Geometry, computation, and interpretation of feature map tensors in CNNs
- [Padding and Stride](cnn/padding_stride.md) -- Controlling output dimensions and downsampling behavior with padding and stride
- [Pooling Layers](cnn/pooling.md) -- Spatial downsampling via max pooling, average pooling, and their role in building hierarchical representations
- [Receptive Field](cnn/receptive_field.md) -- Mathematical analysis of how much spatial context each neuron can access
- [1D Convolutions](cnn/conv1d.md) -- Applying convolutions to sequential data such as time series, audio, and text
- [Dilated Convolutions](cnn/dilated_convolutions.md) -- Expanding receptive fields without increasing parameters using atrous convolution
- [Grouped and Depthwise Separable Convolution](cnn/depthwise_separable.md) -- Efficient convolution factorizations used in MobileNet, EfficientNet, and ShuffleNet
- [Transposed Convolutions](cnn/transposed_conv.md) -- Learnable upsampling for encoder-decoder architectures, GANs, and super-resolution

## 6.2 Residual Connections

Skip connections and residual learning that enable training of very deep networks by providing direct gradient pathways.

- [Residual Connections Overview](residual/residual_connections_overview.md) -- Comprehensive educational module on residual connections and ResNets
- [Skip Connections](residual/skip_connections.md) -- The degradation problem and how shortcut connections solve it
- [Identity Mapping](residual/identity_mapping.md) -- Pre-activation block design for pure identity shortcuts and cleaner gradient flow
- [Gradient Flow Analysis](residual/gradient_flow.md) -- Rigorous mathematical analysis of gradient propagation with and without skip connections
- [ResNet Architecture](residual/resnet.md) -- The ResNet family of architectures from ResNet-18 to ResNet-152 with basic and bottleneck blocks
- [Dense Connections](residual/densenet.md) -- DenseNet's concatenative dense connectivity for maximal feature reuse
- [Highway Networks](residual/highway.md) -- Gated information flow with learned transform and carry gates
- [Exercises](residual/exercises.md) -- Hands-on exercises for implementing and experimenting with residual connections

## 6.3 Vision Transformers

Applying transformer architectures to vision tasks by treating images as sequences of patches.

- [CNN Limitations](vit/cnn_limitations.md) -- Fundamental architectural constraints of CNNs that motivate alternative approaches
- [Bridge: From CNNs to Vision Transformers](vit/cnn_to_vit_bridge.md) -- The evolutionary path from locality-based to attention-based processing
- [Attention Mechanisms in CNNs](vit/attention_in_cnns.md) -- Channel and spatial attention modules (SE, CBAM) that augment CNNs
- [Squeeze-and-Excitation Networks](vit/squeeze_excitation.md) -- Adaptive channel recalibration with minimal computational overhead
- [Non-Local Neural Networks](vit/non_local.md) -- Computing long-range dependencies directly without stacked convolutions
- [Global Average Pooling](vit/global_avg_pooling.md) -- Spatial aggregation as an alternative to fully connected layers
- [ViT Overview](vit/vit_overview.md) -- Introduction to Vision Transformers and the paradigm shift from convolutions to patch sequences
- [Patch Embedding](vit/patch_embedding.md) -- Converting 2D images into 1D sequences of token embeddings
- [Position Embeddings for Images](vit/position_embeddings.md) -- Encoding 2D spatial relationships for permutation-equivariant transformers
- [CLS Token](vit/cls_token.md) -- The learnable classification token mechanism for image-level representation
- [ViT Architecture](vit/vit_architecture.md) -- Complete end-to-end Vision Transformer architecture and scaling behavior
- [DeiT](vit/deit.md) -- Data-efficient Image Transformers that achieve competitive results with only ImageNet-1K
- [Swin Transformer](vit/swin.md) -- Hierarchical vision transformer with shifted window attention and linear complexity
- [Hybrid Architectures](vit/hybrid.md) -- Combining convolutional and transformer components for the best of both worlds
>>>>>>> 96f31bd (...)
