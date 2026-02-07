# Real-World Robustness

## Introduction

Translating adversarial attacks and defenses from digital experiments to the physical world introduces fundamental challenges. Real-world adversarial robustness must account for environmental variability, sensor noise, and the practical constraints of deploying perturbations outside controlled digital settings.

## Digital-to-Physical Gap

### Why Digital Attacks Don't Directly Transfer

Adversarial perturbations crafted in the digital domain often fail when deployed physically due to:

1. **Camera processing**: Auto-exposure, white balance, JPEG compression, and lens distortion alter pixel values
2. **Viewing conditions**: Distance, angle, lighting, and occlusion change the effective perturbation
3. **Print artifacts**: Printer resolution limits, color gamut constraints, and paper reflectance modify the perturbation
4. **Environmental noise**: Weather, atmospheric conditions, and sensor noise add uncontrolled variation

### Expectation over Transformations (EOT)

The standard approach to bridge this gap is optimizing over a distribution of physical transformations:

$$
\boldsymbol{\delta}^* = \arg\max_{\boldsymbol{\delta}} \mathbb{E}_{t \sim \mathcal{T}} \left[ \mathcal{L}(f(t(\mathbf{x} + \boldsymbol{\delta})), y) \right]
$$

where $\mathcal{T}$ includes rotations, scaling, color shifts, noise, and perspective transforms.

## Demonstrated Physical Attacks

### Traffic Sign Attacks

Eykholt et al. (2018) demonstrated physically printed perturbations on stop signs that caused misclassification by autonomous driving classifiers:

- Perturbations were printed as stickers
- Effective from multiple distances and angles
- Survived rain and varying lighting conditions

### Adversarial Objects

Athalye et al. (2018) 3D-printed adversarial objects (e.g., a turtle classified as a rifle) that maintained adversarial properties across viewpoints, demonstrating that physical-world attacks are not merely theoretical.

### Adversarial T-Shirts

Xu et al. (2020) created adversarial patterns printed on clothing that could evade person detection systems, with implications for surveillance and privacy.

## Robustness in Deployment

### Environmental Robustness Testing

For deployed ML systems, robustness should be evaluated against realistic perturbations:

| Perturbation Type | Digital Simulation | Physical Test |
|-------------------|-------------------|---------------|
| Lighting variation | Brightness/contrast augmentation | Multiple lighting setups |
| Camera angle | Affine transforms | Multi-camera evaluation |
| Weather | Synthetic fog/rain overlays | Outdoor testing |
| Sensor noise | Gaussian/salt-and-pepper noise | Different camera hardware |
| Distance | Downsampling | Varying distance capture |

### Common Corruptions Benchmark

Hendrycks & Dietterich (2019) introduced a benchmark of 15 common image corruptions at 5 severity levels, including noise, blur, weather, and digital artifacts. This provides a standardized evaluation of robustness to non-adversarial but realistic perturbations.

## Financial Applications

Real-world robustness concerns for financial ML systems:

- **Data pipeline robustness**: Models must handle missing data, delayed feeds, format changes, and data provider switches without catastrophic failure
- **Distribution shift**: Market regime changes represent a natural form of "real-world perturbation" that models must withstand
- **Sensor reliability**: Alternative data sources (satellite imagery, web scraping, IoT sensors) introduce physical-world noise into financial models
- **Adversarial market participants**: Unlike computer vision where adversarial examples are theoretical, financial markets contain genuinely adversarial actors who manipulate observable signals

## Summary

Real-world robustness extends adversarial robustness from mathematical $\ell_p$ balls to the messy reality of physical deployment. For financial applications, this means considering not just norm-bounded perturbations but the full range of data quality issues, distribution shifts, and strategic adversaries that production systems encounter.

## References

1. Eykholt, K., et al. (2018). "Robust Physical-World Attacks on Deep Learning Visual Classification." CVPR.
2. Athalye, A., et al. (2018). "Synthesizing Robust Adversarial Examples." ICML.
3. Hendrycks, D., & Dietterich, T. (2019). "Benchmarking Neural Network Robustness to Common Corruptions and Perturbations." ICLR.
4. Kurakin, A., Goodfellow, I., & Bengio, S. (2017). "Adversarial Examples in the Physical World." ICLR Workshop.
