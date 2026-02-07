# Self-Supervised Learning in GANs

Self-supervised learning introduces auxiliary tasks that help GANs learn more robust features without requiring labeled data. This approach improves training stability, sample quality, and diversity.

## Motivation

### Problems Self-Supervision Addresses

1. **Mode Collapse**: By learning additional structure, generators maintain diversity
2. **Training Instability**: Auxiliary losses provide more stable gradients
3. **Feature Quality**: Discriminators learn more meaningful representations
4. **Data Efficiency**: Better use of unlabeled data

## Rotation-Based Self-Supervision

### Self-Supervised GAN (SS-GAN)

The discriminator predicts image rotations in addition to real/fake:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SSGANDiscriminator(nn.Module):
    """
    Self-Supervised GAN Discriminator.
    
    Predicts both real/fake and rotation angle (0°, 90°, 180°, 270°).
    """
    
    def __init__(self, image_channels=3, feature_maps=64):
        super().__init__()
        
        ndf = feature_maps
        
        # Shared feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(image_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Real/Fake head
        self.adv_head = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        )
        
        # Rotation prediction head (4 classes)
        self.rot_head = nn.Sequential(
            nn.Conv2d(ndf * 8, ndf * 4, 4, 1, 0, bias=False),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(ndf * 4, 4)
        )
    
    def forward(self, x, return_features=False):
        features = self.features(x)
        
        # Adversarial output
        adv_out = self.adv_head(features).view(-1, 1)
        
        # Rotation prediction
        rot_out = self.rot_head(features)
        
        if return_features:
            return adv_out, rot_out, features
        return adv_out, rot_out


def rotate_batch(images, k):
    """
    Rotate images by k * 90 degrees.
    
    Args:
        images: Tensor (B, C, H, W)
        k: Number of 90-degree rotations
    
    Returns:
        Rotated images
    """
    return torch.rot90(images, k=k, dims=[2, 3])


def create_rotation_batch(images):
    """
    Create batch with all four rotations.
    
    Args:
        images: Original images (B, C, H, W)
    
    Returns:
        rotated_images: (4B, C, H, W)
        rotation_labels: (4B,)
    """
    batch_size = images.size(0)
    
    rotated_images = []
    rotation_labels = []
    
    for k in range(4):
        rotated = rotate_batch(images, k)
        rotated_images.append(rotated)
        rotation_labels.append(torch.full((batch_size,), k, dtype=torch.long))
    
    rotated_images = torch.cat(rotated_images, dim=0)
    rotation_labels = torch.cat(rotation_labels, dim=0)
    
    return rotated_images, rotation_labels


def train_ssgan_step(G, D, real_images, z, optimizer_G, optimizer_D, 
                     criterion_adv, criterion_rot, lambda_rot=1.0):
    """
    Single training step for SS-GAN.
    
    Args:
        G: Generator
        D: Self-supervised discriminator
        real_images: Real images batch
        z: Latent vectors
        optimizer_G, optimizer_D: Optimizers
        criterion_adv: Adversarial loss (BCE)
        criterion_rot: Rotation loss (CrossEntropy)
        lambda_rot: Weight for rotation loss
    
    Returns:
        Dictionary of losses
    """
    batch_size = real_images.size(0)
    device = real_images.device
    
    # Labels
    real_label = torch.ones(batch_size * 4, 1, device=device)
    fake_label = torch.zeros(batch_size * 4, 1, device=device)
    
    # =========================================================================
    # Train Discriminator
    # =========================================================================
    optimizer_D.zero_grad()
    
    # Real images with all rotations
    real_rotated, rot_labels = create_rotation_batch(real_images)
    rot_labels = rot_labels.to(device)
    
    # Forward pass on real
    adv_real, rot_pred_real = D(real_rotated)
    
    # Adversarial loss on real
    loss_adv_real = criterion_adv(torch.sigmoid(adv_real), real_label)
    
    # Rotation loss on real
    loss_rot_real = criterion_rot(rot_pred_real, rot_labels)
    
    # Generate fake images
    fake_images = G(z)
    fake_rotated, rot_labels_fake = create_rotation_batch(fake_images.detach())
    rot_labels_fake = rot_labels_fake.to(device)
    
    # Forward pass on fake
    adv_fake, rot_pred_fake = D(fake_rotated)
    
    # Adversarial loss on fake
    loss_adv_fake = criterion_adv(torch.sigmoid(adv_fake), fake_label)
    
    # Rotation loss on fake
    loss_rot_fake = criterion_rot(rot_pred_fake, rot_labels_fake)
    
    # Total discriminator loss
    loss_D = loss_adv_real + loss_adv_fake + lambda_rot * (loss_rot_real + loss_rot_fake)
    
    loss_D.backward()
    optimizer_D.step()
    
    # =========================================================================
    # Train Generator
    # =========================================================================
    optimizer_G.zero_grad()
    
    fake_images = G(z)
    fake_rotated, rot_labels_fake = create_rotation_batch(fake_images)
    rot_labels_fake = rot_labels_fake.to(device)
    
    adv_fake, rot_pred_fake = D(fake_rotated)
    
    loss_G_adv = criterion_adv(torch.sigmoid(adv_fake), real_label)
    loss_G_rot = criterion_rot(rot_pred_fake, rot_labels_fake)
    
    loss_G = loss_G_adv + lambda_rot * loss_G_rot
    
    loss_G.backward()
    optimizer_G.step()
    
    return {
        'D_adv_real': loss_adv_real.item(),
        'D_adv_fake': loss_adv_fake.item(),
        'D_rot_real': loss_rot_real.item(),
        'D_rot_fake': loss_rot_fake.item(),
        'G_adv': loss_G_adv.item(),
        'G_rot': loss_G_rot.item(),
    }
```

## Contrastive Learning in GANs

### ContraGAN

ContraGAN uses contrastive learning to condition the discriminator:

```python
class ContrastiveDiscriminator(nn.Module):
    """
    Discriminator with contrastive learning for class conditioning.
    """
    
    def __init__(self, image_channels=3, num_classes=10, feature_dim=128):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(image_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        
        # Projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(256, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # Class embeddings
        self.class_embed = nn.Embedding(num_classes, feature_dim)
        
        # Adversarial head
        self.adv_head = nn.Linear(256, 1)
    
    def forward(self, x, labels=None):
        features = self.features(x)
        
        # Adversarial output
        adv_out = self.adv_head(features)
        
        if labels is not None:
            # Contrastive features
            proj = F.normalize(self.projection(features), dim=1)
            class_emb = F.normalize(self.class_embed(labels), dim=1)
            
            return adv_out, proj, class_emb
        
        return adv_out


def contrastive_loss(proj, class_emb, labels, temperature=0.1):
    """
    Compute contrastive loss between projections and class embeddings.
    
    Args:
        proj: Projected features (B, D)
        class_emb: Class embeddings (B, D)
        labels: Class labels (B,)
        temperature: Temperature for softmax
    
    Returns:
        Contrastive loss
    """
    batch_size = proj.size(0)
    
    # Similarity between projections and class embeddings
    # proj: (B, D), class_emb: (B, D)
    sim_matrix = torch.mm(proj, class_emb.t()) / temperature  # (B, B)
    
    # Positive pairs: diagonal elements (same sample)
    # We want proj[i] to be similar to class_emb[i]
    
    # Labels for contrastive loss
    contrastive_labels = torch.arange(batch_size, device=proj.device)
    
    loss = F.cross_entropy(sim_matrix, contrastive_labels)
    
    return loss
```

## Auxiliary Classification GAN (AC-GAN) with Self-Supervision

```python
class ACGANGenerator(nn.Module):
    """Generator conditioned on class labels."""
    
    def __init__(self, latent_dim=100, num_classes=10, feature_maps=64, image_channels=3):
        super().__init__()
        
        self.label_embed = nn.Embedding(num_classes, latent_dim)
        
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim * 2, feature_maps * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(feature_maps, image_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, z, labels):
        label_emb = self.label_embed(labels)
        x = torch.cat([z, label_emb], dim=1)
        x = x.view(x.size(0), -1, 1, 1)
        return self.main(x)


class ACGANDiscriminator(nn.Module):
    """
    Discriminator that predicts real/fake, class, and rotation.
    """
    
    def __init__(self, image_channels=3, num_classes=10, feature_maps=64):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(image_channels, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Real/Fake head
        self.adv_head = nn.Conv2d(feature_maps * 8, 1, 4, 1, 0, bias=False)
        
        # Class prediction head
        self.class_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_maps * 8 * 4 * 4, num_classes)
        )
        
        # Rotation prediction head (self-supervision)
        self.rot_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_maps * 8 * 4 * 4, 4)
        )
    
    def forward(self, x):
        features = self.features(x)
        
        adv_out = self.adv_head(features).view(-1, 1)
        class_out = self.class_head(features)
        rot_out = self.rot_head(features)
        
        return adv_out, class_out, rot_out
```

## Feature Matching with Self-Supervision

```python
class FeatureMatchingSSGAN:
    """
    Self-supervised GAN with feature matching loss.
    """
    
    def __init__(self, G, D, device='cpu'):
        self.G = G
        self.D = D
        self.device = device
    
    def feature_matching_loss(self, real_features, fake_features):
        """
        Compute feature matching loss.
        
        Encourages fake features to match statistics of real features.
        """
        real_mean = real_features.mean(dim=0)
        fake_mean = fake_features.mean(dim=0)
        
        return F.mse_loss(fake_mean, real_mean.detach())
    
    def train_step(self, real_images, z, labels, optimizer_G, optimizer_D):
        batch_size = real_images.size(0)
        
        # =====================================================================
        # Train Discriminator
        # =====================================================================
        optimizer_D.zero_grad()
        
        # Real images
        adv_real, class_real, _, real_features = self.D(real_images)
        
        # Create rotated batch for self-supervision
        real_rotated, rot_labels = create_rotation_batch(real_images)
        rot_labels = rot_labels.to(self.device)
        _, _, rot_pred_real, _ = self.D(real_rotated)
        
        # Fake images
        fake_images = self.G(z, labels)
        adv_fake, class_fake, _, fake_features = self.D(fake_images.detach())
        
        # Losses
        loss_adv = F.binary_cross_entropy_with_logits(
            adv_real, torch.ones_like(adv_real)
        ) + F.binary_cross_entropy_with_logits(
            adv_fake, torch.zeros_like(adv_fake)
        )
        
        loss_class = F.cross_entropy(class_real, labels)
        loss_rot = F.cross_entropy(rot_pred_real, rot_labels)
        
        loss_D = loss_adv + loss_class + 0.5 * loss_rot
        loss_D.backward()
        optimizer_D.step()
        
        # =====================================================================
        # Train Generator
        # =====================================================================
        optimizer_G.zero_grad()
        
        fake_images = self.G(z, labels)
        adv_fake, class_fake, _, fake_features = self.D(fake_images)
        
        # Real features for matching (recompute without gradient)
        with torch.no_grad():
            _, _, _, real_features = self.D(real_images)
        
        # Losses
        loss_G_adv = F.binary_cross_entropy_with_logits(
            adv_fake, torch.ones_like(adv_fake)
        )
        loss_G_class = F.cross_entropy(class_fake, labels)
        loss_G_fm = self.feature_matching_loss(real_features, fake_features)
        
        loss_G = loss_G_adv + loss_G_class + 10.0 * loss_G_fm
        loss_G.backward()
        optimizer_G.step()
        
        return {
            'D_loss': loss_D.item(),
            'G_loss': loss_G.item(),
            'FM_loss': loss_G_fm.item(),
        }
```

## Benefits of Self-Supervision

| Benefit | Mechanism |
|---------|-----------|
| **Improved stability** | Additional gradient signal prevents vanishing gradients |
| **Better features** | Rotation task forces learning of semantic features |
| **Reduced mode collapse** | Multiple objectives maintain diversity |
| **Data efficiency** | No additional labels required |

## Summary

Self-supervised learning enhances GANs by introducing auxiliary tasks that improve feature learning without labeled data. Key approaches include rotation prediction, contrastive learning, and feature matching. These techniques address common GAN challenges like mode collapse and training instability.
