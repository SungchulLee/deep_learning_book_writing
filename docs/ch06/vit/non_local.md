# Non-Local Neural Networks

## Introduction

Non-local operations generalize the concept of local convolutions by computing responses as weighted averages over all spatial and temporal locations in feature maps. By explicitly capturing long-range dependencies without relying on stacked convolutions, non-local networks address a fundamental limitation of convolutional architectures: their inability to efficiently model global context with reasonable depth.

The motivation for non-local operations stems from recognition that many vision tasks—from object detection to semantic segmentation—require understanding relationships between distant image regions. Non-local mechanisms provide a mechanism to compute such relationships directly without requiring progressively enlarged receptive fields through deep stacking.

## Key Concepts

- **Long-Range Dependencies**: Direct computation of feature relationships across entire feature maps
- **Attention-Like Mechanisms**: Learned weighting of feature contributions from all locations
- **Computational Efficiency**: Feasible alternatives to O(HW) stacked convolutions
- **Video Understanding**: Particularly effective for capturing temporal relationships
- **Generality**: Applicable to both spatial and temporal dimensions

## Mathematical Framework

### Non-Local Operations

A non-local operation computes the response at position $\mathbf{x}_i$ as:

$$\mathbf{y}_i = \frac{1}{C(x)} \sum_{j} f(\mathbf{x}_i, \mathbf{x}_j) \cdot g(\mathbf{x}_j)$$

where:
- $f(\mathbf{x}_i, \mathbf{x}_j)$ computes pairwise relationships
- $g(\mathbf{x}_j)$ projects features
- $C(x)$ is a normalization factor

### Pairwise Relation Functions

Common instantiations include:

**Dot-Product (Scaled Attention)**:
$$f(\mathbf{x}_i, \mathbf{x}_j) = \theta(\mathbf{x}_i)^T \phi(\mathbf{x}_j)$$

**Gaussian (RBF Kernel)**:
$$f(\mathbf{x}_i, \mathbf{x}_j) = e^{\theta(\mathbf{x}_i)^T \phi(\mathbf{x}_j)}$$

**Concatenation-Based**:
$$f(\mathbf{x}_i, \mathbf{x}_j) = \text{ReLU}(\mathbf{w}^T[\theta(\mathbf{x}_i) \oplus \phi(\mathbf{x}_j)])$$

where $\oplus$ denotes concatenation.

## Implementation Architecture

### Building Block Structure

A non-local block typically includes:

$$\text{NL}(\mathbf{X}) = \text{Conv}_{1\times 1}(\text{Attention}(\mathbf{X})) + \mathbf{X}$$

where attention is computed via:

$$\text{Attention}(\mathbf{X})_{i} = \text{softmax}(f(\mathbf{x}_i, \{\mathbf{x}_j\})) \cdot g(\mathbf{X})$$

### Computational Complexity

Computing full pairwise interactions requires $O((HW)^2 \times C)$ operations:

$$\text{Flops} = HW \times HW \times C = (HW)^2 C$$

For memory-efficient variants, spatial downsampling reduces complexity:

$$\text{Flops}_{\text{downsampled}} = HW \times \frac{H'W'}{r^2} \times C$$

where $r$ is the downsampling rate.

## Advantages Over Stacked Convolutions

!!! tip "Depth vs. Receptive Field"
    A single non-local operation achieves global receptive field equivalent to O(log HW) convolutional layers, enabling shallower architectures with better gradient flow.

**Direct Relationship Modeling**: Non-local operations directly model relationships between distant locations without intermediate layers.

**Learnable Relevance**: Unlike convolution which uses fixed kernels, non-local operations learn task-specific importance weights.

**Symmetry Properties**: Non-local operations are naturally symmetric in the locations compared, contrasting with asymmetric convolutions.

## Applications in Computer Vision

### Video Understanding

Non-local networks excel at capturing temporal relationships:

$$\text{Response}_{t,i,j} = \sum_{t',i',j'} w(t,i,j,t',i',j') \cdot \mathbf{x}_{t',i',j'}$$

enabling models to recognize actions that depend on temporal correlations.

### Fine-Grained Recognition

For tasks requiring subtle spatial relationships, non-local mechanisms identify discriminative inter-region connections.

### 3D Point Cloud Processing

Non-local operations naturally extend to unstructured point cloud data through learned metric spaces.

## Integration with CNNs

Modern architectures integrate non-local blocks strategically:

- **Early Layers**: Preserve computational efficiency with local convolutions
- **Mid Layers**: Insert non-local blocks to capture semantic relationships
- **Later Layers**: Focus on classification with semantic features

## Computational Optimization Strategies

**Bottleneck Design**: Reduce channel dimension before computing pairwise interactions:

$$\text{Cost} = HW \times HW \times \frac{C}{r} \ll HW \times HW \times C$$

**Spatial Downsampling**: Compute attention on downsampled feature maps.

**Grouped Non-Local**: Apply non-local operations within spatial regions before combining.

## Related Topics

- Attention Mechanisms in CNNs (Chapter 6.1.1)
- Vision Transformer Architecture (Chapter 6.2)
- Temporal Modeling in Videos (Chapter 16)
