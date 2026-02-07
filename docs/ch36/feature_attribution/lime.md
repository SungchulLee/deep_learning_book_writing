# LIME: Local Interpretable Model-agnostic Explanations

## Introduction

LIME (Local Interpretable Model-agnostic Explanations) is a model-agnostic technique that explains individual predictions by approximating the black-box model locally with an interpretable model. The key insight is that even if a model is globally complex, it can be approximated by a simple model in the neighborhood of any specific prediction.

LIME answers the question: **Why did the model make this specific prediction for this specific input?**

## Core Principle

### Local Fidelity

A complex model $f$ may be impossible to interpret globally, but any function can be approximated linearly in a sufficiently small region. LIME exploits this by:

1. Generating perturbed samples around the input
2. Getting the black-box model's predictions for these samples
3. Fitting an interpretable model (e.g., linear regression) to approximate $f$ locally
4. Using the interpretable model's coefficients as explanations

### Mathematical Formulation

LIME finds an explanation $g \in G$ (where $G$ is a class of interpretable models) by minimizing:

$$\xi(x) = \arg\min_{g \in G} \mathcal{L}(f, g, \pi_x) + \Omega(g)$$

where:
- $\mathcal{L}(f, g, \pi_x)$ measures how unfaithful $g$ is to $f$ in the locality defined by $\pi_x$
- $\pi_x$ is a proximity measure defining the neighborhood of $x$
- $\Omega(g)$ is a complexity penalty (e.g., number of features in a linear model)

### Weighted Least Squares

For linear explanations, LIME minimizes:

$$\mathcal{L}(f, g, \pi_x) = \sum_{z \in \mathcal{Z}} \pi_x(z) (f(z) - g(z'))^2$$

where:
- $\mathcal{Z}$ is the set of perturbed samples
- $z'$ is the interpretable representation of $z$
- $\pi_x(z)$ is a weight based on proximity to $x$

## Algorithm

```
Algorithm: LIME
Input: Black-box model f, input x, number of samples N, 
       interpretable model class G, neighborhood kernel π

1. Generate N perturbed samples around x
   For images: randomly mask superpixels
   For text: randomly remove words
   For tabular: perturb feature values

2. For each perturbed sample z:
   a. Get prediction: f(z)
   b. Compute proximity weight: π(x, z)
   c. Convert to interpretable representation: z'

3. Fit weighted linear model:
   g = argmin_{g ∈ G} Σ π(x, z)(f(z) - g(z'))² + Ω(g)

4. Return: coefficients of g as feature importances
```

## PyTorch Implementation

### LIME for Tabular Data

```python
import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import Ridge
from typing import Callable, List, Optional

class LIMETabular:
    """
    LIME for tabular data.
    """
    
    def __init__(
        self,
        model: Callable,
        feature_names: List[str],
        categorical_features: List[int] = None,
        kernel_width: float = None
    ):
        """
        Args:
            model: Prediction function (numpy array -> predictions)
            feature_names: Names of features
            categorical_features: Indices of categorical features
            kernel_width: Width of exponential kernel
        """
        self.model = model
        self.feature_names = feature_names
        self.categorical_features = categorical_features or []
        self.kernel_width = kernel_width
    
    def _perturb_samples(
        self,
        instance: np.ndarray,
        num_samples: int,
        training_data: np.ndarray = None
    ) -> np.ndarray:
        """Generate perturbed samples around instance."""
        n_features = len(instance)
        
        # Sample from normal distribution for continuous features
        perturbations = np.random.normal(0, 1, (num_samples, n_features))
        
        if training_data is not None:
            # Scale by feature standard deviations
            std = training_data.std(axis=0)
            std[std == 0] = 1  # Avoid division by zero
            perturbations *= std
        
        samples = instance + perturbations
        
        # For categorical features, sample from training distribution
        for idx in self.categorical_features:
            if training_data is not None:
                unique_vals = np.unique(training_data[:, idx])
                samples[:, idx] = np.random.choice(
                    unique_vals, size=num_samples
                )
        
        return samples
    
    def _kernel_fn(
        self,
        distances: np.ndarray,
        kernel_width: float
    ) -> np.ndarray:
        """Exponential kernel for weighting samples."""
        return np.sqrt(np.exp(-(distances ** 2) / kernel_width ** 2))
    
    def explain(
        self,
        instance: np.ndarray,
        num_samples: int = 1000,
        num_features: int = 10,
        training_data: np.ndarray = None
    ) -> dict:
        """
        Generate LIME explanation for a single instance.
        
        Args:
            instance: Input to explain (1D array)
            num_samples: Number of perturbed samples
            num_features: Number of features in explanation
            training_data: Training data for scaling
            
        Returns:
            Dictionary with feature importances and local model
        """
        # Generate perturbed samples
        samples = self._perturb_samples(instance, num_samples, training_data)
        
        # Add original instance
        samples = np.vstack([instance.reshape(1, -1), samples])
        
        # Get predictions from black-box model
        predictions = self.model(samples)
        
        # Compute distances from original instance
        distances = np.sqrt(((samples - instance) ** 2).sum(axis=1))
        
        # Compute kernel weights
        if self.kernel_width is None:
            self.kernel_width = np.sqrt(len(instance)) * 0.75
        weights = self._kernel_fn(distances, self.kernel_width)
        
        # Fit weighted Ridge regression
        explainer = Ridge(alpha=1.0, fit_intercept=True)
        explainer.fit(samples, predictions, sample_weight=weights)
        
        # Get feature importances
        importances = explainer.coef_
        
        # Sort by absolute importance
        sorted_idx = np.argsort(np.abs(importances))[::-1]
        top_features = sorted_idx[:num_features]
        
        explanation = {
            'feature_importance': dict(zip(
                [self.feature_names[i] for i in top_features],
                importances[top_features]
            )),
            'intercept': explainer.intercept_,
            'local_pred': explainer.predict(instance.reshape(1, -1))[0],
            'actual_pred': predictions[0],
            'r2_score': explainer.score(samples, predictions, sample_weight=weights)
        }
        
        return explanation


def visualize_lime_explanation(explanation: dict, figsize=(10, 6)):
    """Visualize LIME feature importances."""
    import matplotlib.pyplot as plt
    
    features = list(explanation['feature_importance'].keys())
    importances = list(explanation['feature_importance'].values())
    
    colors = ['green' if v > 0 else 'red' for v in importances]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    y_pos = np.arange(len(features))
    ax.barh(y_pos, importances, color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.set_xlabel('Feature Importance')
    ax.set_title(f'LIME Explanation (Local R² = {explanation["r2_score"]:.3f})')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    return fig
```

### LIME for Images

```python
from skimage.segmentation import quickshift
import cv2

class LIMEImage:
    """
    LIME for image classification.
    """
    
    def __init__(
        self,
        model: Callable,
        num_classes: int
    ):
        """
        Args:
            model: Image classifier (batch of images -> class probabilities)
            num_classes: Number of classes
        """
        self.model = model
        self.num_classes = num_classes
    
    def _segment_image(
        self,
        image: np.ndarray,
        kernel_size: int = 4,
        max_dist: int = 200,
        ratio: float = 0.2
    ) -> np.ndarray:
        """Segment image into superpixels."""
        segments = quickshift(
            image, 
            kernel_size=kernel_size, 
            max_dist=max_dist, 
            ratio=ratio
        )
        return segments
    
    def _perturb_image(
        self,
        image: np.ndarray,
        segments: np.ndarray,
        num_samples: int,
        hide_color: tuple = None
    ) -> tuple:
        """Generate perturbed images by masking superpixels."""
        n_segments = np.unique(segments).shape[0]
        
        # Random binary masks for segments
        masks = np.random.randint(0, 2, (num_samples, n_segments))
        
        # Always include original (all segments on)
        masks[0] = 1
        
        # Hide color (gray by default)
        if hide_color is None:
            hide_color = np.mean(image, axis=(0, 1))
        
        # Generate perturbed images
        perturbed_images = []
        for mask in masks:
            perturbed = image.copy()
            for seg_idx, active in enumerate(mask):
                if not active:
                    perturbed[segments == seg_idx] = hide_color
            perturbed_images.append(perturbed)
        
        return np.array(perturbed_images), masks
    
    def explain(
        self,
        image: np.ndarray,
        target_class: int,
        num_samples: int = 1000,
        num_features: int = 10
    ) -> dict:
        """
        Generate LIME explanation for image.
        
        Args:
            image: Input image (H, W, 3) normalized to [0, 1]
            target_class: Class to explain
            num_samples: Number of perturbed samples
            num_features: Number of superpixels in explanation
            
        Returns:
            Explanation dictionary
        """
        # Segment image
        segments = self._segment_image(image)
        n_segments = np.unique(segments).shape[0]
        
        # Generate perturbed images
        perturbed_images, masks = self._perturb_image(
            image, segments, num_samples
        )
        
        # Get model predictions
        predictions = self.model(perturbed_images)
        target_probs = predictions[:, target_class]
        
        # Compute distances (number of active superpixels)
        distances = np.sqrt((1 - masks).sum(axis=1))
        
        # Kernel weights
        kernel_width = 0.25 * np.sqrt(n_segments)
        weights = np.exp(-(distances ** 2) / (kernel_width ** 2))
        
        # Fit weighted linear model
        explainer = Ridge(alpha=1.0)
        explainer.fit(masks, target_probs, sample_weight=weights)
        
        # Get superpixel importances
        importances = explainer.coef_
        
        # Sort by importance
        sorted_idx = np.argsort(importances)[::-1]
        top_segments = sorted_idx[:num_features]
        
        explanation = {
            'segments': segments,
            'segment_importance': importances,
            'top_segments': top_segments,
            'local_pred': explainer.predict(masks[0:1])[0],
            'actual_pred': target_probs[0]
        }
        
        return explanation
    
    def visualize_explanation(
        self,
        image: np.ndarray,
        explanation: dict,
        positive_only: bool = True,
        num_features: int = 5
    ) -> np.ndarray:
        """Create visualization highlighting important regions."""
        segments = explanation['segments']
        importances = explanation['segment_importance']
        
        # Create mask for top features
        if positive_only:
            sorted_idx = np.argsort(importances)[::-1]
        else:
            sorted_idx = np.argsort(np.abs(importances))[::-1]
        
        top_segments = sorted_idx[:num_features]
        
        # Create visualization
        mask = np.zeros_like(segments, dtype=float)
        for seg_idx in top_segments:
            mask[segments == seg_idx] = importances[seg_idx]
        
        # Normalize mask
        if mask.max() > 0:
            mask = mask / mask.max()
        
        # Create overlay
        heatmap = cv2.applyColorMap(
            np.uint8(255 * mask), 
            cv2.COLORMAP_JET
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        
        visualization = 0.6 * image + 0.4 * heatmap
        visualization = np.clip(visualization, 0, 1)
        
        return visualization
```

## Complete Example

```python
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

def lime_image_example():
    """Complete LIME example for image classification."""
    
    # Load pre-trained model
    model = models.resnet50(pretrained=True)
    model.eval()
    
    # Preprocessing
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Model wrapper for LIME
    def predict_fn(images):
        """Convert numpy images to predictions."""
        batch = []
        for img in images:
            # Convert from [0, 1] to tensor with normalization
            pil_img = Image.fromarray((img * 255).astype(np.uint8))
            tensor = preprocess(pil_img)
            batch.append(tensor)
        
        batch = torch.stack(batch)
        
        with torch.no_grad():
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)
        
        return probs.numpy()
    
    # Load and process image
    image = Image.open('example.jpg').convert('RGB')
    image_np = np.array(image.resize((224, 224))) / 255.0
    
    # Get prediction
    probs = predict_fn(image_np[np.newaxis])
    pred_class = probs.argmax()
    pred_prob = probs[0, pred_class]
    
    print(f"Predicted class: {pred_class}, Probability: {pred_prob:.3f}")
    
    # LIME explanation
    lime = LIMEImage(predict_fn, num_classes=1000)
    explanation = lime.explain(
        image_np,
        target_class=pred_class,
        num_samples=1000,
        num_features=10
    )
    
    # Visualize
    vis = lime.visualize_explanation(image_np, explanation, num_features=5)
    
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image_np)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(explanation['segments'], cmap='nipy_spectral')
    axes[1].set_title('Superpixels')
    axes[1].axis('off')
    
    axes[2].imshow(vis)
    axes[2].set_title(f'LIME Explanation (class {pred_class})')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('lime_explanation.png', dpi=150)
```

## Advantages and Limitations

### Advantages

1. **Model-agnostic**: Works with any black-box model
2. **Interpretable**: Produces human-readable explanations
3. **Flexible**: Applicable to various data types
4. **Local fidelity**: Accurate in the neighborhood of the instance

### Limitations

1. **Sampling variance**: Different runs may produce different explanations
2. **Kernel width**: Results sensitive to kernel width choice
3. **Superpixel quality**: Image explanations depend on segmentation
4. **Linear assumption**: May not capture non-linear local behavior

## Applications in Finance

### Credit Risk Explanation

```python
def explain_credit_decision(
    model,
    applicant_features,
    feature_names,
    training_data
):
    """
    Explain why a credit application was approved/denied.
    """
    def predict_fn(X):
        with torch.no_grad():
            tensor = torch.tensor(X, dtype=torch.float32)
            probs = torch.sigmoid(model(tensor)).numpy()
        return probs
    
    lime = LIMETabular(
        model=predict_fn,
        feature_names=feature_names
    )
    
    explanation = lime.explain(
        applicant_features,
        num_samples=5000,
        num_features=10,
        training_data=training_data
    )
    
    print("\n=== Credit Decision Explanation ===")
    print(f"Predicted probability: {explanation['actual_pred']:.3f}")
    print("\nTop contributing factors:")
    
    for feature, importance in explanation['feature_importance'].items():
        direction = "↑ approval" if importance > 0 else "↓ approval"
        print(f"  {feature}: {importance:+.4f} ({direction})")
    
    return explanation
```

## References

1. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?: Explaining the Predictions of Any Classifier." KDD.

2. Garreau, D., & von Luxburg, U. (2020). "Explaining the Explainer: A First Theoretical Analysis of LIME." AISTATS.

3. Slack, D., et al. (2020). "Fooling LIME and SHAP: Adversarial Attacks on Post hoc Explanation Methods." AIES.
