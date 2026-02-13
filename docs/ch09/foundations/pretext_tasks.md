# Pretext Tasks in Self-Supervised Learning

## Learning Objectives

By the end of this section, you will be able to:

- Understand the fundamental concept of pretext tasks and their role in self-supervised learning
- Differentiate between various categories of pretext tasks
- Implement common pretext tasks in PyTorch
- Analyze the inductive biases introduced by different pretext task designs
- Select appropriate pretext tasks for specific downstream applications

## Introduction

Self-supervised learning (SSL) has emerged as a powerful paradigm for learning meaningful representations from unlabeled data. At the heart of SSL lies the concept of **pretext tasks**—auxiliary learning objectives that create supervision signals directly from the input data structure, without requiring human annotations.

### The Core Idea

The fundamental insight behind pretext tasks is that certain properties of data can be predicted from other properties of the same data. By training a neural network to solve these prediction problems, the network learns representations that capture semantic information useful for downstream tasks.

$$\mathcal{L}_{\text{pretext}} = \mathbb{E}_{x \sim \mathcal{D}} \left[ \ell(f_\theta(T(x)), y(x)) \right]$$

where:
- $x$ is the input data
- $T(x)$ is a transformation applied to create the pretext task input
- $y(x)$ is the pseudo-label derived from $x$ itself
- $f_\theta$ is the neural network with parameters $\theta$
- $\ell$ is the loss function

## Taxonomy of Pretext Tasks

### 1. Context-Based Pretext Tasks

Context-based methods exploit spatial or temporal relationships within data.

#### Relative Position Prediction

Given image patches, predict their relative spatial arrangement:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

class RelativePositionTask(nn.Module):
    """
    Pretext task: Predict relative position of two patches.
    Given a 3x3 grid, classify which of 8 positions the second 
    patch occupies relative to the center patch.
    """
    def __init__(self, backbone, patch_size=96, gap=48):
        super().__init__()
        self.backbone = backbone
        self.patch_size = patch_size
        self.gap = gap  # Gap between patches to increase difficulty
        
        # Assuming backbone outputs feature_dim dimensional features
        feature_dim = 512  # Adjust based on your backbone
        
        # Classifier for 8 relative positions
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 8)  # 8 possible positions
        )
        
        # Position offsets: (row_offset, col_offset) for 8 neighbors
        self.position_offsets = [
            (-1, -1), (-1, 0), (-1, 1),  # Top row
            (0, -1),          (0, 1),     # Middle row (excluding center)
            (1, -1),  (1, 0),  (1, 1)     # Bottom row
        ]
    
    def extract_patches(self, image, center_pos, neighbor_idx):
        """
        Extract center patch and one of its 8 neighbors.
        
        Args:
            image: Input image tensor (C, H, W)
            center_pos: (row, col) position of center in the grid
            neighbor_idx: Index of the neighbor (0-7)
        
        Returns:
            center_patch, neighbor_patch: Two extracted patches
        """
        p = self.patch_size
        g = self.gap
        
        # Center patch coordinates
        cy, cx = center_pos[0] * (p + g), center_pos[1] * (p + g)
        center_patch = image[:, cy:cy+p, cx:cx+p]
        
        # Neighbor patch coordinates
        offset = self.position_offsets[neighbor_idx]
        ny = cy + offset[0] * (p + g)
        nx = cx + offset[1] * (p + g)
        neighbor_patch = image[:, ny:ny+p, nx:nx+p]
        
        return center_patch, neighbor_patch
    
    def forward(self, patch1, patch2):
        """
        Forward pass: predict relative position.
        
        Args:
            patch1: Center patch (B, C, H, W)
            patch2: Neighbor patch (B, C, H, W)
        
        Returns:
            logits: Position predictions (B, 8)
        """
        # Extract features from both patches
        feat1 = self.backbone(patch1)
        feat2 = self.backbone(patch2)
        
        # Flatten if needed
        if len(feat1.shape) > 2:
            feat1 = F.adaptive_avg_pool2d(feat1, 1).flatten(1)
            feat2 = F.adaptive_avg_pool2d(feat2, 1).flatten(1)
        
        # Concatenate features and classify
        combined = torch.cat([feat1, feat2], dim=1)
        logits = self.classifier(combined)
        
        return logits


def create_position_dataset(images, patch_size=96, gap=48):
    """
    Create dataset for relative position prediction task.
    
    Args:
        images: List of PIL Images or tensor batch
        patch_size: Size of each patch
        gap: Gap between patches
    
    Returns:
        List of (patch1, patch2, label) tuples
    """
    dataset = []
    grid_size = 3
    required_size = grid_size * patch_size + (grid_size - 1) * gap
    
    transform = transforms.Compose([
        transforms.Resize((required_size, required_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    for img in images:
        if isinstance(img, Image.Image):
            img_tensor = transform(img)
        else:
            img_tensor = img
        
        # Center position (1, 1) in 3x3 grid
        center_y = patch_size + gap
        center_x = patch_size + gap
        
        # Extract center patch
        center_patch = img_tensor[:, center_y:center_y+patch_size, 
                                  center_x:center_x+patch_size]
        
        # Extract each neighbor and create samples
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), 
                   (0, 1), (1, -1), (1, 0), (1, 1)]
        
        for label, (dy, dx) in enumerate(offsets):
            ny = center_y + dy * (patch_size + gap)
            nx = center_x + dx * (patch_size + gap)
            neighbor_patch = img_tensor[:, ny:ny+patch_size, nx:nx+patch_size]
            
            dataset.append((center_patch.clone(), neighbor_patch.clone(), label))
    
    return dataset
```

#### Jigsaw Puzzle Solving

Reconstruct the original arrangement of shuffled image patches:

```python
import itertools

class JigsawPuzzleTask(nn.Module):
    """
    Jigsaw Puzzle pretext task.
    Predict the permutation used to shuffle a 3x3 grid of patches.
    
    Uses a predefined set of permutations (typically 100-1000) rather
    than all 9! = 362880 possible permutations.
    """
    def __init__(self, backbone, num_permutations=100, patch_size=64):
        super().__init__()
        self.backbone = backbone
        self.patch_size = patch_size
        self.num_patches = 9  # 3x3 grid
        
        # Generate a fixed set of permutations
        # Using maximal hamming distance selection for diversity
        self.permutations = self._generate_permutations(num_permutations)
        
        feature_dim = 512  # Adjust based on backbone
        
        # Context-Free Network (CFN) approach: 
        # Process each patch independently, then combine
        self.fc = nn.Sequential(
            nn.Linear(feature_dim * self.num_patches, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_permutations)
        )
    
    def _generate_permutations(self, num_perms):
        """
        Generate diverse permutations using hamming distance criterion.
        """
        # Start with all possible permutations
        all_perms = list(itertools.permutations(range(9)))
        
        if num_perms >= len(all_perms):
            return torch.tensor(all_perms)
        
        # Greedy selection based on hamming distance
        selected = [all_perms[0]]
        all_perms = all_perms[1:]
        
        while len(selected) < num_perms:
            # Find permutation with maximum minimum hamming distance
            best_perm = None
            best_min_dist = -1
            
            for perm in all_perms:
                min_dist = min(
                    sum(p1 != p2 for p1, p2 in zip(perm, sel))
                    for sel in selected
                )
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_perm = perm
            
            if best_perm is not None:
                selected.append(best_perm)
                all_perms.remove(best_perm)
        
        return torch.tensor(selected)
    
    def extract_patches(self, image):
        """
        Extract 9 patches from a 3x3 grid.
        
        Args:
            image: Input image (C, H, W)
        
        Returns:
            patches: List of 9 patches
        """
        p = self.patch_size
        patches = []
        
        for i in range(3):
            for j in range(3):
                patch = image[:, i*p:(i+1)*p, j*p:(j+1)*p]
                patches.append(patch)
        
        return patches
    
    def shuffle_patches(self, patches, perm_idx):
        """
        Shuffle patches according to a permutation.
        
        Args:
            patches: List of 9 patches
            perm_idx: Index into self.permutations
        
        Returns:
            shuffled_patches: Shuffled list of patches
        """
        perm = self.permutations[perm_idx]
        return [patches[i] for i in perm]
    
    def forward(self, patches_batch):
        """
        Forward pass for jigsaw puzzle solving.
        
        Args:
            patches_batch: (B, 9, C, H, W) - batch of shuffled patch sets
        
        Returns:
            logits: (B, num_permutations) - permutation predictions
        """
        B = patches_batch.shape[0]
        
        # Process each patch through the backbone
        features = []
        for i in range(self.num_patches):
            patch = patches_batch[:, i]  # (B, C, H, W)
            feat = self.backbone(patch)
            if len(feat.shape) > 2:
                feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
            features.append(feat)
        
        # Concatenate all patch features
        combined = torch.cat(features, dim=1)  # (B, feature_dim * 9)
        
        # Classify the permutation
        logits = self.fc(combined)
        
        return logits


def jigsaw_collate_fn(batch, model, num_permutations):
    """
    Custom collate function for jigsaw puzzle training.
    
    Args:
        batch: List of images
        model: JigsawPuzzleTask model (for permutations)
        num_permutations: Number of possible permutations
    
    Returns:
        patches_batch: (B, 9, C, H, W)
        labels: (B,) permutation indices
    """
    batch_patches = []
    labels = []
    
    for img in batch:
        # Extract patches
        patches = model.extract_patches(img)
        
        # Random permutation index
        perm_idx = torch.randint(0, num_permutations, (1,)).item()
        
        # Shuffle patches
        shuffled = model.shuffle_patches(patches, perm_idx)
        
        batch_patches.append(torch.stack(shuffled))
        labels.append(perm_idx)
    
    return torch.stack(batch_patches), torch.tensor(labels)
```

### 2. Generation-Based Pretext Tasks

These tasks require the model to reconstruct or generate parts of the input.

#### Image Inpainting

Predict missing regions of an image:

```python
class InpaintingTask(nn.Module):
    """
    Context Encoder: Learning visual features by inpainting.
    Reconstruct masked regions of images.
    """
    def __init__(self, encoder, decoder, mask_size=64):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mask_size = mask_size
    
    def create_mask(self, batch_size, img_size, device):
        """
        Create random rectangular masks.
        
        Args:
            batch_size: Number of masks to create
            img_size: Size of the image (H, W)
            device: Device for tensor creation
        
        Returns:
            masks: Binary masks (B, 1, H, W), 1 = keep, 0 = mask
            mask_regions: List of (y, x, h, w) tuples
        """
        H, W = img_size
        masks = torch.ones(batch_size, 1, H, W, device=device)
        mask_regions = []
        
        for i in range(batch_size):
            # Random position for mask
            y = torch.randint(0, H - self.mask_size, (1,)).item()
            x = torch.randint(0, W - self.mask_size, (1,)).item()
            
            masks[i, :, y:y+self.mask_size, x:x+self.mask_size] = 0
            mask_regions.append((y, x, self.mask_size, self.mask_size))
        
        return masks, mask_regions
    
    def forward(self, images):
        """
        Forward pass for inpainting.
        
        Args:
            images: Input images (B, C, H, W)
        
        Returns:
            reconstructed: Reconstructed images
            masks: Applied masks
            mask_regions: Masked region coordinates
        """
        B, C, H, W = images.shape
        
        # Create masks
        masks, mask_regions = self.create_mask(B, (H, W), images.device)
        
        # Apply masks to input (mask out regions)
        masked_images = images * masks
        
        # Encode masked images
        features = self.encoder(masked_images)
        
        # Decode to reconstruct
        reconstructed = self.decoder(features)
        
        return reconstructed, masks, mask_regions
    
    def compute_loss(self, images, reconstructed, masks):
        """
        Compute inpainting loss (only on masked regions).
        
        Uses combination of L2 reconstruction loss and adversarial loss.
        
        Args:
            images: Original images
            reconstructed: Reconstructed images
            masks: Binary masks (1 = keep, 0 = masked)
        
        Returns:
            loss: Reconstruction loss on masked regions
        """
        # Invert mask: 1 = masked region, 0 = visible region
        inv_masks = 1 - masks
        
        # L2 loss on masked regions
        reconstruction_loss = F.mse_loss(
            reconstructed * inv_masks,
            images * inv_masks,
            reduction='sum'
        ) / inv_masks.sum()
        
        return reconstruction_loss


class ContextEncoderDecoder(nn.Module):
    """
    Full Context Encoder architecture for inpainting.
    """
    def __init__(self, in_channels=3):
        super().__init__()
        
        # Encoder: downsample to bottleneck
        self.encoder = nn.Sequential(
            # 128 -> 64
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 -> 32
            nn.Conv2d(64, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # 32 -> 16
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 16 -> 8
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 8 -> 4
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Channel-wise fully connected (bottleneck)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 4000, 4),  # 4 -> 1
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(4000, 512, 4),  # 1 -> 4
            nn.ReLU(inplace=True),
        )
        
        # Decoder: upsample back to original size
        self.decoder = nn.Sequential(
            # 4 -> 8
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 8 -> 16
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 16 -> 32
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 32 -> 64
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 64 -> 128
            nn.ConvTranspose2d(64, in_channels, 4, stride=2, padding=1),
            nn.Tanh(),
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        bottleneck = self.bottleneck(encoded)
        decoded = self.decoder(bottleneck)
        return decoded
```

#### Colorization

Predict color from grayscale images:

```python
class ColorizationTask(nn.Module):
    """
    Colorful Image Colorization pretext task.
    Predict ab channels in LAB color space from L channel.
    """
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        
        # Decoder for colorization
        # Output: 313 bins (quantized ab space)
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Upsample(scale_factor=2),
            
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Upsample(scale_factor=2),
            
            nn.Conv2d(64, 313, 1),  # 313 quantized ab bins
        )
        
        # Load ab quantization (predefined bins)
        self.register_buffer('ab_bins', self._create_ab_bins())
    
    def _create_ab_bins(self, grid_size=10):
        """
        Create quantized ab color bins.
        
        Returns:
            ab_bins: (313, 2) tensor of ab values
        """
        # Create grid in ab space
        # ab values typically range from -128 to 127
        a_range = torch.linspace(-110, 110, grid_size * 2)
        b_range = torch.linspace(-110, 110, grid_size * 2)
        
        # Create meshgrid
        a_grid, b_grid = torch.meshgrid(a_range, b_range, indexing='ij')
        
        # Flatten and filter to gamut
        ab_bins = torch.stack([a_grid.flatten(), b_grid.flatten()], dim=1)
        
        # Filter to valid gamut (simplified: keep points within certain radius)
        valid_mask = (ab_bins[:, 0]**2 + ab_bins[:, 1]**2) < 110**2
        ab_bins = ab_bins[valid_mask]
        
        # Subsample to exactly 313 bins if needed
        if len(ab_bins) > 313:
            indices = torch.linspace(0, len(ab_bins)-1, 313).long()
            ab_bins = ab_bins[indices]
        
        return ab_bins
    
    def rgb_to_lab(self, rgb):
        """
        Convert RGB image to LAB color space.
        Simplified conversion (for production, use skimage or cv2).
        
        Args:
            rgb: (B, 3, H, W) RGB image [0, 1]
        
        Returns:
            L: (B, 1, H, W) Lightness channel [0, 100]
            ab: (B, 2, H, W) ab channels [-128, 127]
        """
        # Simplified conversion - in practice use proper color conversion
        # This is a placeholder showing the interface
        r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        
        # Approximate L channel (luminance)
        L = 0.2126 * r + 0.7152 * g + 0.0722 * b
        L = L.unsqueeze(1) * 100  # Scale to [0, 100]
        
        # Approximate ab (simplified)
        a = (r - g) * 127
        b_ch = (b - (r + g) / 2) * 127
        ab = torch.stack([a, b_ch], dim=1)
        
        return L, ab
    
    def quantize_ab(self, ab):
        """
        Quantize ab values to nearest bins.
        
        Args:
            ab: (B, 2, H, W) continuous ab values
        
        Returns:
            indices: (B, H, W) bin indices
        """
        B, _, H, W = ab.shape
        ab_flat = ab.permute(0, 2, 3, 1).reshape(-1, 2)  # (B*H*W, 2)
        
        # Find nearest bin for each pixel
        distances = torch.cdist(ab_flat, self.ab_bins)  # (B*H*W, 313)
        indices = distances.argmin(dim=1)  # (B*H*W,)
        
        return indices.reshape(B, H, W)
    
    def forward(self, L):
        """
        Forward pass: predict ab from L.
        
        Args:
            L: Grayscale/Lightness channel (B, 1, H, W)
        
        Returns:
            logits: (B, 313, H, W) bin probabilities
        """
        # Encode grayscale image
        features = self.encoder(L.repeat(1, 3, 1, 1))  # Repeat L to 3 channels
        
        # Decode to ab predictions
        logits = self.decoder(features)
        
        return logits
    
    def compute_loss(self, logits, ab_targets):
        """
        Compute cross-entropy loss with class rebalancing.
        
        Args:
            logits: (B, 313, H, W) predicted bin probabilities
            ab_targets: (B, 2, H, W) target ab values
        
        Returns:
            loss: Cross-entropy loss
        """
        # Quantize targets
        target_indices = self.quantize_ab(ab_targets)  # (B, H, W)
        
        # Cross-entropy loss
        loss = F.cross_entropy(logits, target_indices)
        
        return loss
```

### 3. Transformation Prediction Tasks

Predict the transformation applied to the input.

#### Rotation Prediction

```python
class RotationPredictionTask(nn.Module):
    """
    Unsupervised Representation Learning by Predicting Image Rotations.
    Predict which of 4 rotations (0°, 90°, 180°, 270°) was applied.
    """
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        
        # Rotation angles
        self.rotations = [0, 90, 180, 270]
        
        feature_dim = 512  # Adjust based on backbone
        self.classifier = nn.Linear(feature_dim, 4)
    
    def rotate_image(self, image, angle_idx):
        """
        Rotate image by specified angle.
        
        Args:
            image: (C, H, W) or (B, C, H, W)
            angle_idx: 0=0°, 1=90°, 2=180°, 3=270°
        
        Returns:
            Rotated image
        """
        if angle_idx == 0:
            return image
        elif angle_idx == 1:
            return torch.rot90(image, k=1, dims=[-2, -1])
        elif angle_idx == 2:
            return torch.rot90(image, k=2, dims=[-2, -1])
        elif angle_idx == 3:
            return torch.rot90(image, k=3, dims=[-2, -1])
    
    def forward(self, images):
        """
        Forward pass: predict rotation.
        
        Args:
            images: Rotated images (B, C, H, W)
        
        Returns:
            logits: Rotation predictions (B, 4)
        """
        features = self.backbone(images)
        
        if len(features.shape) > 2:
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        
        logits = self.classifier(features)
        return logits
    
    def create_rotated_batch(self, images):
        """
        Create a batch with all rotations of each image.
        
        Args:
            images: Original images (B, C, H, W)
        
        Returns:
            rotated_images: (B*4, C, H, W)
            labels: (B*4,) rotation indices
        """
        B = images.shape[0]
        rotated = []
        labels = []
        
        for i in range(4):
            rotated.append(self.rotate_image(images, i))
            labels.extend([i] * B)
        
        rotated_images = torch.cat(rotated, dim=0)
        labels = torch.tensor(labels, device=images.device)
        
        return rotated_images, labels


def train_rotation_epoch(model, dataloader, optimizer, device):
    """
    Train rotation prediction for one epoch.
    
    Args:
        model: RotationPredictionTask model
        dataloader: Data loader (images only, no labels needed)
        optimizer: Optimizer
        device: Device
    
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    for images, _ in dataloader:  # Ignore original labels
        images = images.to(device)
        
        # Create rotated batch
        rotated_images, labels = model.create_rotated_batch(images)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(rotated_images)
        
        # Compute loss
        loss = F.cross_entropy(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item() * len(labels)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += len(labels)
    
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples * 100
    
    print(f"Rotation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    return avg_loss
```

### 4. Temporal Order Prediction (for Video)

```python
class TemporalOrderTask(nn.Module):
    """
    Shuffle and Learn: Temporal order verification for video.
    Determine if video frames are in correct temporal order.
    """
    def __init__(self, backbone, num_frames=3):
        super().__init__()
        self.backbone = backbone
        self.num_frames = num_frames
        
        feature_dim = 512
        
        # Process frame features and predict order
        self.temporal_classifier = nn.Sequential(
            nn.Linear(feature_dim * num_frames, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 2)  # Binary: correct order or not
        )
    
    def forward(self, frames):
        """
        Forward pass for temporal order verification.
        
        Args:
            frames: (B, num_frames, C, H, W) video frames
        
        Returns:
            logits: (B, 2) order prediction
        """
        B, T, C, H, W = frames.shape
        
        # Process each frame
        features = []
        for t in range(T):
            frame = frames[:, t]
            feat = self.backbone(frame)
            if len(feat.shape) > 2:
                feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
            features.append(feat)
        
        # Concatenate temporal features
        combined = torch.cat(features, dim=1)
        
        # Classify order
        logits = self.temporal_classifier(combined)
        
        return logits
    
    def create_training_samples(self, video_frames):
        """
        Create positive (correct order) and negative (shuffled) samples.
        
        Args:
            video_frames: (B, T, C, H, W) video frames
        
        Returns:
            samples: (B*2, num_frames, C, H, W)
            labels: (B*2,) binary labels
        """
        B = video_frames.shape[0]
        
        positive_samples = video_frames.clone()
        negative_samples = video_frames.clone()
        
        # Shuffle negative samples
        for i in range(B):
            # Random permutation
            perm = torch.randperm(self.num_frames)
            while torch.all(perm == torch.arange(self.num_frames)):
                perm = torch.randperm(self.num_frames)
            negative_samples[i] = video_frames[i, perm]
        
        samples = torch.cat([positive_samples, negative_samples], dim=0)
        labels = torch.cat([
            torch.ones(B),   # Positive: correct order
            torch.zeros(B)   # Negative: shuffled
        ]).long()
        
        return samples, labels
```

## Comparison of Pretext Tasks

| Task | Input Modification | Prediction Target | Best For | Limitations |
|------|-------------------|-------------------|----------|-------------|
| Relative Position | Extract patches | Position (8 classes) | Spatial reasoning | Ignores global context |
| Jigsaw Puzzle | Shuffle patches | Permutation class | Spatial structure | Computationally expensive |
| Rotation | Rotate image | Angle (4 classes) | Geometric features | Limited to rotation-invariant |
| Colorization | Remove color | Color (ab bins) | Semantic features | Color-invariant features weak |
| Inpainting | Mask regions | Pixel reconstruction | Generative modeling | High compute for reconstruction |
| Temporal Order | Sample frames | Order (binary) | Video understanding | Requires temporal data |

## Inductive Biases and Feature Quality

Different pretext tasks induce different inductive biases:

```python
def analyze_learned_features(model, dataloader, device, task_type):
    """
    Analyze the quality of learned features for different pretext tasks.
    
    Args:
        model: Trained pretext task model
        dataloader: Evaluation data loader
        device: Device
        task_type: Type of pretext task
    
    Returns:
        Feature statistics and transferability metrics
    """
    model.eval()
    features_list = []
    labels_list = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            
            # Extract features (before task-specific head)
            if hasattr(model, 'backbone'):
                features = model.backbone(images)
            elif hasattr(model, 'encoder'):
                features = model.encoder(images)
            else:
                features = model(images)
            
            if len(features.shape) > 2:
                features = F.adaptive_avg_pool2d(features, 1).flatten(1)
            
            features_list.append(features.cpu())
            labels_list.append(labels)
    
    features = torch.cat(features_list, dim=0).numpy()
    labels = torch.cat(labels_list, dim=0).numpy()
    
    # Compute statistics
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    
    # Normalize features
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)
    
    # Linear evaluation (measures transferability)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    scores = cross_val_score(clf, features_normalized, labels, cv=5)
    
    # Feature statistics
    feature_stats = {
        'task_type': task_type,
        'feature_dim': features.shape[1],
        'feature_mean': features.mean(),
        'feature_std': features.std(),
        'linear_accuracy': scores.mean() * 100,
        'linear_accuracy_std': scores.std() * 100,
    }
    
    print(f"\n{task_type} Feature Analysis:")
    print(f"  Feature dimension: {feature_stats['feature_dim']}")
    print(f"  Linear evaluation accuracy: {feature_stats['linear_accuracy']:.2f}% "
          f"(±{feature_stats['linear_accuracy_std']:.2f}%)")
    
    return feature_stats
```

## Best Practices

### 1. Task Selection Guidelines

```python
def select_pretext_task(data_characteristics, downstream_task):
    """
    Guidelines for selecting appropriate pretext tasks.
    
    Args:
        data_characteristics: Dict with data properties
        downstream_task: Target downstream application
    
    Returns:
        Recommended pretext task(s)
    """
    recommendations = []
    
    # Check data modality
    if data_characteristics.get('modality') == 'image':
        # For image classification downstream
        if downstream_task == 'classification':
            recommendations.append({
                'task': 'rotation_prediction',
                'reason': 'Learns semantic features invariant to rotation',
                'compute': 'low'
            })
            recommendations.append({
                'task': 'contrastive_learning',
                'reason': 'Best overall performance, especially with SimCLR/MoCo',
                'compute': 'medium-high'
            })
        
        # For object detection/segmentation
        elif downstream_task == 'detection':
            recommendations.append({
                'task': 'jigsaw_puzzle',
                'reason': 'Learns spatial relationships crucial for localization',
                'compute': 'medium'
            })
            recommendations.append({
                'task': 'inpainting',
                'reason': 'Learns contextual reasoning for object boundaries',
                'compute': 'high'
            })
    
    elif data_characteristics.get('modality') == 'video':
        recommendations.append({
            'task': 'temporal_order',
            'reason': 'Captures temporal dynamics essential for video',
            'compute': 'medium'
        })
    
    return recommendations
```

### 2. Training Tips

```python
# Recommended hyperparameters for common pretext tasks
PRETEXT_CONFIGS = {
    'rotation': {
        'batch_size': 256,
        'lr': 0.1,
        'optimizer': 'SGD',
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'epochs': 100,
        'lr_schedule': 'cosine'
    },
    'jigsaw': {
        'batch_size': 128,
        'lr': 0.01,
        'optimizer': 'SGD',
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'epochs': 70,
        'lr_schedule': 'step',
        'lr_milestones': [30, 50]
    },
    'colorization': {
        'batch_size': 64,
        'lr': 3.16e-5,
        'optimizer': 'Adam',
        'betas': (0.9, 0.99),
        'epochs': 450000,  # Iterations
        'lr_schedule': None
    },
    'inpainting': {
        'batch_size': 64,
        'lr': 2e-4,
        'optimizer': 'Adam',
        'betas': (0.5, 0.999),
        'epochs': 100000,  # Iterations
        'adversarial_weight': 0.001
    }
}
```

## Summary

Pretext tasks form the foundation of self-supervised learning by creating supervision from the data itself. Key insights:

1. **Context-based tasks** (relative position, jigsaw) learn spatial relationships
2. **Generation-based tasks** (inpainting, colorization) learn semantic representations
3. **Transformation prediction** (rotation) learns geometric invariances
4. **Temporal tasks** extend SSL to video data

Modern contrastive learning methods (covered in subsequent sections) have largely superseded these classical pretext tasks for image representation learning, but understanding pretext tasks provides crucial intuition for the SSL paradigm.

## References

1. Doersch, C., Gupta, A., & Efros, A. A. (2015). Unsupervised visual representation learning by context prediction. ICCV.
2. Noroozi, M., & Favaro, P. (2016). Unsupervised learning of visual representations by solving jigsaw puzzles. ECCV.
3. Gidaris, S., Singh, P., & Komodakis, N. (2018). Unsupervised representation learning by predicting image rotations. ICLR.
4. Zhang, R., Isola, P., & Efros, A. A. (2016). Colorful image colorization. ECCV.
5. Pathak, D., et al. (2016). Context encoders: Feature learning by inpainting. CVPR.
6. Misra, I., Zitnick, C. L., & Hebert, M. (2016). Shuffle and learn: Unsupervised learning using temporal order verification. ECCV.
