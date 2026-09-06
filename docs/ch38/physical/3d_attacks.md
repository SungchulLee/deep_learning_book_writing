# 3D Adversarial Attacks
## Introduction

**3D adversarial attacks** extend adversarial perturbations beyond the image plane into three-dimensional space. Instead of perturbing pixel values, these attacks modify the physical properties of objects—their shape, texture, or lighting—to create adversarial objects that fool classifiers from multiple viewpoints.

## Motivation

Standard $\ell_p$ perturbations in image space have a fundamental limitation: they are **view-dependent**. A perturbation crafted for one camera angle becomes invalid when the viewpoint changes. 3D attacks address this by optimizing perturbations that are adversarial across a distribution of viewpoints.

## Mathematical Formulation

### Rendering Pipeline

A 3D object with parameters $\theta_{3D}$ (mesh, texture, pose) is projected to a 2D image via a differentiable renderer $\mathcal{R}$:

$$
\mathbf{I} = \mathcal{R}(\theta_{3D}, \theta_{\text{cam}}, \theta_{\text{light}})
$$

### 3D Adversarial Optimization

The attack optimizes the 3D perturbation to be adversarial across viewpoints:

$$
\boldsymbol{\delta}_{3D}^* = \arg\max_{\boldsymbol{\delta}_{3D}} \mathbb{E}_{\theta_{\text{cam}}, \theta_{\text{light}}} \left[ \mathcal{L}(f(\mathcal{R}(\theta_{3D} + \boldsymbol{\delta}_{3D}, \theta_{\text{cam}}, \theta_{\text{light}})), y) \right]
$$

The perturbation $\boldsymbol{\delta}_{3D}$ can modify:

- **Texture**: Adversarial textures applied to object surfaces
- **Shape**: Small mesh vertex displacements
- **Material**: Changes to surface reflectance properties

### Differentiable Rendering

The key enabling technology is **differentiable rendering**, which allows gradients to flow from the 2D classification loss back through the rendering pipeline to 3D object parameters:

$$
\frac{\partial \mathcal{L}}{\partial \boldsymbol{\delta}_{3D}} = \frac{\partial \mathcal{L}}{\partial \mathbf{I}} \cdot \frac{\partial \mathcal{R}}{\partial \boldsymbol{\delta}_{3D}}
$$

## Attack Types

### Adversarial Textures

Modify the texture map of a 3D object so that renderings from any angle are adversarial:

$$
T^* = \arg\max_T \mathbb{E}_v \left[ \mathcal{L}(f(\mathcal{R}(\text{mesh}, T, v)), y) \right]
$$

### Adversarial Shapes

Subtly deform the object mesh while maintaining recognizability to humans:

$$
V^* = \arg\max_{\|V' - V\| \leq \varepsilon} \mathbb{E}_v \left[ \mathcal{L}(f(\mathcal{R}(V', T, v)), y) \right]
$$

### Adversarial Lighting

Manipulate the lighting environment to create adversarial conditions:

$$
L^* = \arg\max_L \mathcal{L}(f(\mathcal{R}(\text{mesh}, T, v, L)), y)
$$

## Practical Implications

### Autonomous Systems

3D adversarial attacks have been demonstrated against:

- **Autonomous vehicles**: Adversarial 3D-printed objects misclassified by perception systems
- **Drone navigation**: Modified landmarks that confuse visual positioning
- **Robotic manipulation**: Adversarial object shapes that cause grasping failures

### Defense Implications

3D attacks motivate defenses that consider:

- Multi-view consistency checking
- 3D-aware feature representations
- Robust perception under environmental variation

## Summary

| Attack Type | Perturbation Space | View-Invariant | Physical Feasibility |
|-------------|-------------------|----------------|---------------------|
| $\ell_p$ image | 2D pixel space | No | Limited |
| Adversarial patch | 2D local region | Partially | High |
| 3D texture | Texture map | Yes | Moderate |
| 3D shape | Mesh vertices | Yes | Requires 3D printing |

3D adversarial attacks represent the most realistic threat model for physical-world perception systems, requiring defenses that go beyond 2D robustness.

## References

1. Athalye, A., et al. (2018). "Synthesizing Robust Adversarial Examples." ICML.
2. Xiao, C., et al. (2019). "MeshAdv: Adversarial Meshes for Visual Recognition." CVPR.
3. Zeng, X., et al. (2019). "Adversarial Attacks Beyond the Image Space." CVPR.

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
