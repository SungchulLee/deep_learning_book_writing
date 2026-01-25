# Data Augmentation

## Overview

Data augmentation is a regularization technique that artificially expands the training dataset by creating modified versions of existing samples. By exposing the model to transformed data that preserves semantic meaning, augmentation teaches invariance to irrelevant variations and significantly improves generalization.

## Conceptual Foundation

### Why Data Augmentation Works

Data augmentation addresses the fundamental problem of limited training data:

1. **Increased effective dataset size**: More training examples without additional labeling
2. **Invariance learning**: Model learns features robust to transformations
3. **Reduced overfitting**: Harder to memorize augmented, varying data
4. **Implicit regularization**: Constrains the hypothesis space to transformation-invariant solutions

### Mathematical Perspective

Augmentation can be viewed as adding a regularization term to the loss. For a transformation $T$ applied with probability $p$:

$$
\mathcal{L}_{\text{aug}} = \mathbb{E}_{x, y \sim \mathcal{D}} \left[ \mathbb{E}_{T \sim \mathcal{T}} \left[ \ell(f(T(x)), y) \right] \right]
$$

This encourages:
$$
f(T(x)) \approx f(x) \quad \forall T \in \mathcal{T}
$$

## Image Augmentation

### Geometric Transformations

```python
import torch
import torchvision.transforms as T
from torchvision.transforms import functional as F
import numpy as np
from PIL import Image

class GeometricAugmentation:
    """Standard geometric augmentations for images."""
    
    def __init__(
        self,
        rotation_range: float = 15,
        translate_range: float = 0.1,
        scale_range: tuple = (0.9, 1.1),
        shear_range: float = 10,
        flip_horizontal: bool = True,
        flip_vertical: bool = False
    ):
        transforms = []
        
        # Random affine (rotation, translation, scale, shear)
        if any([rotation_range, translate_range, scale_range != (1, 1), shear_range]):
            transforms.append(T.RandomAffine(
                degrees=rotation_range,
                translate=(translate_range, translate_range) if translate_range else None,
                scale=scale_range,
                shear=shear_range
            ))
        
        # Flips
        if flip_horizontal:
            transforms.append(T.RandomHorizontalFlip(p=0.5))
        if flip_vertical:
            transforms.append(T.RandomVerticalFlip(p=0.5))
        
        self.transform = T.Compose(transforms)
    
    def __call__(self, image):
        return self.transform(image)


# PyTorch standard geometric transforms
geometric_transforms = T.Compose([
    T.RandomRotation(degrees=15),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1),
        shear=10
    ),
    T.RandomPerspective(distortion_scale=0.2, p=0.5),
    T.RandomResizedCrop(size=224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
])
```

### Color/Photometric Transformations

```python
class PhotometricAugmentation:
    """Color and lighting augmentations."""
    
    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.1
    ):
        self.transform = T.Compose([
            T.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue
            ),
            T.RandomGrayscale(p=0.1),
            T.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
            T.RandomAutocontrast(p=0.3),
            T.RandomEqualize(p=0.3),
        ])
    
    def __call__(self, image):
        return self.transform(image)


# Standard color augmentation
color_transforms = T.Compose([
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    T.RandomGrayscale(p=0.2),
])
```

### Noise and Blur

```python
class NoiseAugmentation:
    """Add various types of noise to images."""
    
    def __init__(self, noise_types=['gaussian', 'blur', 'jpeg']):
        self.noise_types = noise_types
    
    def __call__(self, image):
        if isinstance(image, Image.Image):
            image = T.ToTensor()(image)
        
        noise_type = np.random.choice(self.noise_types)
        
        if noise_type == 'gaussian':
            noise = torch.randn_like(image) * 0.05
            image = torch.clamp(image + noise, 0, 1)
        elif noise_type == 'blur':
            image = T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))(image)
        elif noise_type == 'jpeg':
            # Simulate JPEG compression artifacts
            quality = np.random.randint(30, 95)
            pil_img = T.ToPILImage()(image)
            import io
            buffer = io.BytesIO()
            pil_img.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            image = T.ToTensor()(Image.open(buffer))
        
        return image
```

### Advanced Augmentations

#### Cutout / Random Erasing

```python
class Cutout:
    """
    Randomly mask out square regions of the image.
    
    Reference: DeVries & Taylor, "Improved Regularization of CNNs with Cutout"
    """
    
    def __init__(self, n_holes: int = 1, length: int = 16):
        self.n_holes = n_holes
        self.length = length
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        h, w = img.shape[-2:]
        mask = torch.ones_like(img)
        
        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            
            y1 = max(0, y - self.length // 2)
            y2 = min(h, y + self.length // 2)
            x1 = max(0, x - self.length // 2)
            x2 = min(w, x + self.length // 2)
            
            mask[..., y1:y2, x1:x2] = 0
        
        return img * mask


# PyTorch built-in
random_erasing = T.RandomErasing(
    p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0
)
```

#### Mixup

```python
def mixup_data(x: torch.Tensor, y: torch.Tensor, 
               alpha: float = 0.2) -> tuple:
    """
    Mixup: blend pairs of examples and their labels.
    
    Reference: Zhang et al., "mixup: Beyond Empirical Risk Minimization"
    
    Args:
        x: Input batch
        y: Labels (one-hot or class indices)
        alpha: Beta distribution parameter
        
    Returns:
        mixed_x, y_a, y_b, lam
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute mixup loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# Usage in training loop
for x, y in train_loader:
    x, y_a, y_b, lam = mixup_data(x, y, alpha=0.2)
    outputs = model(x)
    loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
```

#### CutMix

```python
def cutmix_data(x: torch.Tensor, y: torch.Tensor, 
                alpha: float = 1.0) -> tuple:
    """
    CutMix: cut and paste patches between training images.
    
    Reference: Yun et al., "CutMix: Regularization Strategy to Train CNNs"
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    
    # Get bounding box
    _, _, H, W = x.shape
    cut_ratio = np.sqrt(1 - lam)
    cut_w = int(W * cut_ratio)
    cut_h = int(H * cut_ratio)
    
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    
    # Cut and paste
    x_cutmix = x.clone()
    x_cutmix[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    
    # Adjust lambda based on actual box area
    lam = 1 - ((x2 - x1) * (y2 - y1)) / (W * H)
    
    y_a, y_b = y, y[index]
    return x_cutmix, y_a, y_b, lam
```

### Complete Image Augmentation Pipeline

```python
def get_train_transforms(image_size: int = 224, augment_level: str = 'standard'):
    """
    Get training transforms based on augmentation level.
    
    Args:
        image_size: Target image size
        augment_level: 'minimal', 'standard', 'aggressive'
    """
    if augment_level == 'minimal':
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    elif augment_level == 'standard':
        return T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.RandomErasing(p=0.25)
        ])
    
    elif augment_level == 'aggressive':
        return T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.2),
            T.RandomRotation(degrees=30),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            T.RandomGrayscale(p=0.2),
            T.RandomAffine(degrees=0, translate=(0.2, 0.2), shear=15),
            T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.RandomErasing(p=0.5, scale=(0.02, 0.4))
        ])
    
    raise ValueError(f"Unknown augment_level: {augment_level}")


def get_val_transforms(image_size: int = 224):
    """Validation/test transforms (no augmentation)."""
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
```

## Text Augmentation

### Basic Text Augmentations

```python
import random
import nltk
from nltk.corpus import wordnet

class TextAugmentation:
    """Text augmentation techniques."""
    
    def __init__(self, aug_prob: float = 0.3):
        self.aug_prob = aug_prob
        # Download required NLTK data
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
    
    def synonym_replacement(self, text: str, n: int = 1) -> str:
        """Replace n words with synonyms."""
        words = text.split()
        new_words = words.copy()
        
        # Get words that have synonyms
        random_word_list = list(set([w for w in words if self._get_synonyms(w)]))
        random.shuffle(random_word_list)
        
        num_replaced = 0
        for random_word in random_word_list:
            synonyms = self._get_synonyms(random_word)
            if synonyms:
                synonym = random.choice(synonyms)
                new_words = [synonym if w == random_word else w for w in new_words]
                num_replaced += 1
            if num_replaced >= n:
                break
        
        return ' '.join(new_words)
    
    def _get_synonyms(self, word: str) -> list:
        """Get synonyms from WordNet."""
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym.lower() != word.lower():
                    synonyms.add(synonym)
        return list(synonyms)
    
    def random_insertion(self, text: str, n: int = 1) -> str:
        """Randomly insert n synonyms of random words."""
        words = text.split()
        new_words = words.copy()
        
        for _ in range(n):
            word = random.choice(words)
            synonyms = self._get_synonyms(word)
            if synonyms:
                insert_pos = random.randint(0, len(new_words))
                new_words.insert(insert_pos, random.choice(synonyms))
        
        return ' '.join(new_words)
    
    def random_swap(self, text: str, n: int = 1) -> str:
        """Randomly swap n pairs of words."""
        words = text.split()
        new_words = words.copy()
        
        for _ in range(n):
            if len(new_words) >= 2:
                idx1, idx2 = random.sample(range(len(new_words)), 2)
                new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
        
        return ' '.join(new_words)
    
    def random_deletion(self, text: str, p: float = 0.1) -> str:
        """Randomly delete words with probability p."""
        words = text.split()
        if len(words) == 1:
            return text
        
        new_words = [w for w in words if random.random() > p]
        if not new_words:
            return random.choice(words)
        
        return ' '.join(new_words)
    
    def augment(self, text: str) -> str:
        """Apply random augmentation."""
        if random.random() > self.aug_prob:
            return text
        
        aug_type = random.choice(['synonym', 'insert', 'swap', 'delete'])
        
        if aug_type == 'synonym':
            return self.synonym_replacement(text)
        elif aug_type == 'insert':
            return self.random_insertion(text)
        elif aug_type == 'swap':
            return self.random_swap(text)
        else:
            return self.random_deletion(text)
```

### Back-Translation

```python
class BackTranslation:
    """
    Augment text by translating to another language and back.
    Requires transformers library.
    """
    
    def __init__(self, intermediate_lang: str = 'de'):
        from transformers import MarianMTModel, MarianTokenizer
        
        # English to intermediate
        self.en_to_lang = MarianMTModel.from_pretrained(
            f'Helsinki-NLP/opus-mt-en-{intermediate_lang}'
        )
        self.en_to_lang_tokenizer = MarianTokenizer.from_pretrained(
            f'Helsinki-NLP/opus-mt-en-{intermediate_lang}'
        )
        
        # Intermediate to English
        self.lang_to_en = MarianMTModel.from_pretrained(
            f'Helsinki-NLP/opus-mt-{intermediate_lang}-en'
        )
        self.lang_to_en_tokenizer = MarianTokenizer.from_pretrained(
            f'Helsinki-NLP/opus-mt-{intermediate_lang}-en'
        )
    
    def translate(self, text: str, model, tokenizer) -> str:
        inputs = tokenizer(text, return_tensors='pt', padding=True)
        translated = model.generate(**inputs)
        return tokenizer.decode(translated[0], skip_special_tokens=True)
    
    def augment(self, text: str) -> str:
        # Translate to intermediate language
        intermediate = self.translate(text, self.en_to_lang, self.en_to_lang_tokenizer)
        # Translate back to English
        back_translated = self.translate(intermediate, self.lang_to_en, self.lang_to_en_tokenizer)
        return back_translated
```

## Time Series Augmentation

```python
import numpy as np
import torch

class TimeSeriesAugmentation:
    """Augmentations for time series data."""
    
    @staticmethod
    def jittering(x: np.ndarray, sigma: float = 0.03) -> np.ndarray:
        """Add Gaussian noise."""
        return x + np.random.normal(0, sigma, x.shape)
    
    @staticmethod
    def scaling(x: np.ndarray, sigma: float = 0.1) -> np.ndarray:
        """Scale by random factor."""
        factor = np.random.normal(1, sigma, (1, x.shape[1]))
        return x * factor
    
    @staticmethod
    def magnitude_warping(x: np.ndarray, sigma: float = 0.2, 
                          knot: int = 4) -> np.ndarray:
        """Warp magnitude with smooth curve."""
        from scipy.interpolate import CubicSpline
        
        orig_steps = np.arange(x.shape[0])
        random_warps = np.random.normal(1.0, sigma, (knot + 2, x.shape[1]))
        warp_steps = np.linspace(0, x.shape[0] - 1, knot + 2)
        
        warper = np.zeros((x.shape[0], x.shape[1]))
        for i in range(x.shape[1]):
            warper[:, i] = CubicSpline(warp_steps, random_warps[:, i])(orig_steps)
        
        return x * warper
    
    @staticmethod
    def time_warping(x: np.ndarray, sigma: float = 0.2, 
                     knot: int = 4) -> np.ndarray:
        """Warp time axis with smooth curve."""
        from scipy.interpolate import CubicSpline
        
        orig_steps = np.arange(x.shape[0])
        random_warps = np.random.normal(1.0, sigma, knot + 2)
        warp_steps = np.linspace(0, x.shape[0] - 1, knot + 2)
        
        time_warp = CubicSpline(warp_steps, warp_steps * random_warps)(orig_steps)
        time_warp = np.clip(time_warp, 0, x.shape[0] - 1)
        
        warped = np.zeros_like(x)
        for i in range(x.shape[1]):
            warped[:, i] = np.interp(orig_steps, time_warp, x[:, i])
        
        return warped
    
    @staticmethod
    def window_slicing(x: np.ndarray, reduce_ratio: float = 0.9) -> np.ndarray:
        """Take random contiguous slice and resize."""
        target_len = int(x.shape[0] * reduce_ratio)
        if target_len < 1:
            return x
        
        start = np.random.randint(0, x.shape[0] - target_len + 1)
        sliced = x[start:start + target_len]
        
        # Resize back to original length
        indices = np.linspace(0, target_len - 1, x.shape[0])
        resized = np.zeros_like(x)
        for i in range(x.shape[1]):
            resized[:, i] = np.interp(indices, np.arange(target_len), sliced[:, i])
        
        return resized
    
    @staticmethod
    def permutation(x: np.ndarray, max_segments: int = 5) -> np.ndarray:
        """Randomly permute segments of the series."""
        n_segments = np.random.randint(2, max_segments + 1)
        segment_len = x.shape[0] // n_segments
        
        segments = []
        for i in range(n_segments):
            start = i * segment_len
            end = (i + 1) * segment_len if i < n_segments - 1 else x.shape[0]
            segments.append(x[start:end])
        
        np.random.shuffle(segments)
        return np.concatenate(segments, axis=0)
```

## Tabular Data Augmentation

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors

class TabularAugmentation:
    """Augmentation techniques for tabular data."""
    
    @staticmethod
    def add_gaussian_noise(X: np.ndarray, noise_scale: float = 0.1) -> np.ndarray:
        """Add Gaussian noise to features."""
        std = np.std(X, axis=0) * noise_scale
        noise = np.random.normal(0, std, X.shape)
        return X + noise
    
    @staticmethod
    def feature_dropout(X: np.ndarray, dropout_rate: float = 0.1) -> np.ndarray:
        """Randomly set features to their column mean."""
        X_aug = X.copy()
        mask = np.random.random(X.shape) < dropout_rate
        col_means = np.mean(X, axis=0)
        X_aug[mask] = np.tile(col_means, (X.shape[0], 1))[mask]
        return X_aug
    
    @staticmethod
    def smote(X: np.ndarray, y: np.ndarray, 
              minority_class: int = 1,
              k_neighbors: int = 5,
              n_synthetic: int = None) -> tuple:
        """
        SMOTE: Synthetic Minority Over-sampling Technique.
        
        Generates synthetic samples for minority class.
        """
        minority_mask = y == minority_class
        X_minority = X[minority_mask]
        
        if n_synthetic is None:
            majority_count = np.sum(~minority_mask)
            minority_count = np.sum(minority_mask)
            n_synthetic = majority_count - minority_count
        
        if n_synthetic <= 0 or len(X_minority) < k_neighbors:
            return X, y
        
        # Find k nearest neighbors
        nn = NearestNeighbors(n_neighbors=k_neighbors + 1)
        nn.fit(X_minority)
        
        synthetic_samples = []
        for _ in range(n_synthetic):
            # Random minority sample
            idx = np.random.randint(len(X_minority))
            sample = X_minority[idx]
            
            # Random neighbor
            distances, indices = nn.kneighbors([sample])
            neighbor_idx = np.random.choice(indices[0][1:])  # Exclude self
            neighbor = X_minority[neighbor_idx]
            
            # Interpolate
            alpha = np.random.random()
            synthetic = sample + alpha * (neighbor - sample)
            synthetic_samples.append(synthetic)
        
        X_synthetic = np.array(synthetic_samples)
        y_synthetic = np.full(n_synthetic, minority_class)
        
        return np.vstack([X, X_synthetic]), np.concatenate([y, y_synthetic])


class MixupTabular:
    """Mixup for tabular data."""
    
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
    
    def __call__(self, X: np.ndarray, y: np.ndarray) -> tuple:
        n = len(X)
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha, n)
        else:
            lam = np.ones(n)
        
        lam = lam.reshape(-1, 1)
        index = np.random.permutation(n)
        
        X_mixed = lam * X + (1 - lam) * X[index]
        y_mixed = lam.squeeze() * y + (1 - lam.squeeze()) * y[index]
        
        return X_mixed, y_mixed
```

## AutoAugment and Learned Augmentation

```python
class RandAugment:
    """
    RandAugment: simplified learned augmentation policy.
    
    Reference: Cubuk et al., "RandAugment: Practical Automated Data Augmentation"
    """
    
    def __init__(self, n_ops: int = 2, magnitude: int = 9):
        """
        Args:
            n_ops: Number of augmentation operations to apply
            magnitude: Strength of augmentation (0-30)
        """
        self.n_ops = n_ops
        self.magnitude = magnitude
        
        # Define available operations
        self.ops = [
            'identity', 'autocontrast', 'equalize', 'rotate',
            'solarize', 'color', 'posterize', 'contrast',
            'brightness', 'sharpness', 'shear_x', 'shear_y',
            'translate_x', 'translate_y'
        ]
    
    def __call__(self, image):
        # Select random operations
        ops = random.sample(self.ops, self.n_ops)
        
        for op in ops:
            image = self._apply_op(image, op, self.magnitude)
        
        return image
    
    def _apply_op(self, img, op: str, magnitude: int):
        """Apply a single augmentation operation."""
        # Magnitude to actual values
        mag = magnitude / 30.0  # Normalize to 0-1
        
        if op == 'identity':
            return img
        elif op == 'autocontrast':
            return F.autocontrast(img)
        elif op == 'equalize':
            return F.equalize(img)
        elif op == 'rotate':
            angle = mag * 30  # Up to 30 degrees
            return F.rotate(img, angle)
        elif op == 'solarize':
            threshold = int((1 - mag) * 255)
            return F.solarize(img, threshold)
        elif op == 'posterize':
            bits = int(8 - mag * 4)
            return F.posterize(img, bits)
        elif op == 'contrast':
            factor = 1 + mag * 0.9 * random.choice([-1, 1])
            return F.adjust_contrast(img, factor)
        elif op == 'brightness':
            factor = 1 + mag * 0.9 * random.choice([-1, 1])
            return F.adjust_brightness(img, factor)
        elif op == 'sharpness':
            factor = 1 + mag * 0.9 * random.choice([-1, 1])
            return F.adjust_sharpness(img, factor)
        elif op == 'shear_x':
            shear = mag * 0.3 * random.choice([-1, 1])
            return F.affine(img, angle=0, translate=[0, 0], scale=1, shear=[shear, 0])
        elif op == 'shear_y':
            shear = mag * 0.3 * random.choice([-1, 1])
            return F.affine(img, angle=0, translate=[0, 0], scale=1, shear=[0, shear])
        elif op == 'translate_x':
            shift = int(mag * img.size[0] * 0.3) * random.choice([-1, 1])
            return F.affine(img, angle=0, translate=[shift, 0], scale=1, shear=[0, 0])
        elif op == 'translate_y':
            shift = int(mag * img.size[1] * 0.3) * random.choice([-1, 1])
            return F.affine(img, angle=0, translate=[0, shift], scale=1, shear=[0, 0])
        
        return img
```

## Practical Guidelines

### Choosing Augmentations

| Data Type | Recommended Augmentations |
|-----------|--------------------------|
| Natural images | Flips, crops, color jitter, RandAugment |
| Medical images | Rotations, scaling (careful with flips) |
| Documents | Slight rotations, noise, blur |
| Time series | Jittering, scaling, time warping |
| Text | Synonym replacement, back-translation |
| Tabular | Noise injection, SMOTE for imbalance |

### Augmentation Strength

- **Too weak**: Little regularization benefit
- **Too strong**: May destroy semantic information
- **Best practice**: Start moderate, increase if overfitting persists

### Validation Set

**Never augment validation/test data** - use original samples for fair evaluation.

## References

1. Shorten, C., & Khoshgoftaar, T. M. (2019). A Survey on Image Data Augmentation for Deep Learning. *Journal of Big Data*, 6(1), 60.
2. Zhang, H., et al. (2018). mixup: Beyond Empirical Risk Minimization. *ICLR*.
3. DeVries, T., & Taylor, G. W. (2017). Improved Regularization of CNNs with Cutout. *arXiv*.
4. Cubuk, E. D., et al. (2020). RandAugment: Practical Automated Data Augmentation. *NeurIPS*.
