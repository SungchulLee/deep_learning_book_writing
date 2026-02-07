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
