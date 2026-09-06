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

## Exercises

**Exercise 1.**
For a linear classifier $f(x) = w^T x + b$, compute the minimal $\ell_\infty$ perturbation needed to change the predicted class. Relate this to the robustness of neural networks.

??? success "Solution to Exercise 1"
    For a linear classifier, the distance to the decision boundary under $\ell_\infty$ norm is $\frac{|w^T x + b|}{\|w\|_1}$. The minimal perturbation is $\delta^* = \frac{|w^T x + b|}{\|w\|_1} \cdot \text{sign}(w)$. For neural networks, the local linear approximation $f(x + \delta) \approx f(x) + \nabla_x f \cdot \delta$ explains why FGSM (which uses the sign of the gradient) is effective. The vulnerability of high-dimensional models comes from the fact that $\|w\|_1$ grows with dimension while $|w^T x + b|$ does not necessarily, making the robustness margin shrink. $\square$

---

**Exercise 2.**
Implement the attack or defense described in this section for a ResNet-18 model on CIFAR-10. Report clean accuracy and robust accuracy under PGD-20 attack with $\epsilon = 8/255$.

??? success "Solution to Exercise 2"
    A standard ResNet-18 achieves $\sim$93% clean accuracy but $\sim$0% robust accuracy under PGD-20 ($\epsilon = 8/255$, step size $2/255$). After applying the method from this section, typical results depend on the specific technique: adversarial training achieves $\sim$83% clean / $\sim$50% robust; certified defenses achieve lower but provable bounds. The accuracy-robustness trade-off is fundamental: improving robustness typically costs 5--15% clean accuracy. Report results averaged over 3 random seeds with standard errors. $\square$

---

**Exercise 3.**
Prove that no defense can simultaneously achieve high accuracy on clean data and high robustness against $\ell_\infty$ perturbations without increasing model capacity, under the assumption that the data distribution has overlapping class-conditional supports within the perturbation ball.

??? success "Solution to Exercise 3"
    If two classes have support overlap within distance $\epsilon$ (i.e., $\exists x_1 \in \text{class 1}, x_2 \in \text{class 2}$ with $\|x_1 - x_2\|_\infty \leq 2\epsilon$), then any classifier robust at both $x_1$ and $x_2$ must misclassify at least one of them (the perturbation balls overlap). This is the fundamental accuracy-robustness trade-off: the fraction of overlapping support determines the unavoidable accuracy loss. For natural image distributions, significant overlap exists at $\epsilon = 8/255$, explaining the observed 10--15% accuracy drop. Increased model capacity (wider networks) can better approximate the complex robust decision boundary, partially mitigating the trade-off. $\square$

---

**Exercise 4.**
Discuss how adversarial robustness concerns manifest in a financial machine learning system (e.g., fraud detection or trading signal generation). How does the threat model differ from computer vision?

??? success "Solution to Exercise 4"
    In finance, adversaries are strategic agents (fraudsters, market manipulators) who actively adapt to detection systems. Key differences from vision: (1) the perturbation space is constrained by what is economically feasible (a fraudster cannot change their entire transaction history); (2) attacks are sequential and adaptive (the adversary observes the system's response and adjusts); (3) the cost of false positives and false negatives is asymmetric (blocking a legitimate transaction vs. missing fraud); (4) $\ell_p$ norms are not meaningful -- domain-specific perturbation models are needed. Defenses must be robust to adaptive adversaries, which rules out many detection-based approaches that can be evaded once the detection criterion is known. $\square$
