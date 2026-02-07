# Broader GAN Applications

GANs have found transformative applications across numerous domains beyond image generation. This section covers applications in image synthesis, healthcare, and scientific research.

---

# Image Generation and Synthesis Applications

GANs have revolutionized image generation, enabling applications from artistic creation to practical computer vision tasks. This section covers key applications in image synthesis.

## Image Super-Resolution

### SRGAN Architecture

```python
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """Residual block for SRGAN generator."""
    
    def __init__(self, channels):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
        )
    
    def forward(self, x):
        return x + self.block(x)


class SRGANGenerator(nn.Module):
    """
    Super-Resolution GAN Generator.
    
    Upscales images by 4x using residual learning.
    """
    
    def __init__(self, in_channels=3, num_residuals=16, upscale_factor=4):
        super().__init__()
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 9, 1, 4),
            nn.PReLU(),
        )
        
        # Residual blocks
        self.residuals = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_residuals)]
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
        )
        
        # Upsampling blocks
        upsample_blocks = []
        for _ in range(upscale_factor // 2):
            upsample_blocks.extend([
                nn.Conv2d(64, 256, 3, 1, 1),
                nn.PixelShuffle(2),  # Sub-pixel convolution
                nn.PReLU(),
            ])
        self.upsample = nn.Sequential(*upsample_blocks)
        
        # Final convolution
        self.conv3 = nn.Conv2d(64, in_channels, 9, 1, 4)
    
    def forward(self, x):
        out1 = self.conv1(x)
        out = self.residuals(out1)
        out = self.conv2(out) + out1  # Skip connection
        out = self.upsample(out)
        out = self.conv3(out)
        return torch.tanh(out)


class SRGANDiscriminator(nn.Module):
    """Discriminator for super-resolution."""
    
    def __init__(self, in_channels=3):
        super().__init__()
        
        def conv_block(in_c, out_c, stride):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, stride, 1),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=True),
            )
        
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            
            conv_block(64, 64, 2),
            conv_block(64, 128, 1),
            conv_block(128, 128, 2),
            conv_block(128, 256, 1),
            conv_block(256, 256, 2),
            conv_block(256, 512, 1),
            conv_block(512, 512, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
        )
    
    def forward(self, x):
        features = self.main(x)
        return self.classifier(features)


def perceptual_loss(vgg, generated, target):
    """Perceptual loss using VGG features."""
    gen_features = vgg(generated)
    target_features = vgg(target)
    return nn.functional.mse_loss(gen_features, target_features)
```

## Image Inpainting

### Context Encoder for Inpainting

```python
class InpaintingGenerator(nn.Module):
    """
    Generator for image inpainting.
    
    Fills in missing regions based on surrounding context.
    """
    
    def __init__(self, in_channels=4):  # RGB + mask
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )
        
        # Bottleneck with dilated convolutions
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 2, dilation=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 4, dilation=4),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 8, dilation=8),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh(),
        )
    
    def forward(self, image, mask):
        x = torch.cat([image, mask], dim=1)
        encoded = self.encoder(x)
        bottleneck = self.bottleneck(encoded)
        decoded = self.decoder(bottleneck)
        output = image * (1 - mask) + decoded * mask
        return output
```

## Style Transfer

### Fast Neural Style Transfer

```python
class StyleTransferGenerator(nn.Module):
    """Fast neural style transfer generator."""
    
    def __init__(self):
        super().__init__()
        
        # Downsampling
        self.down = nn.Sequential(
            nn.ReflectionPad2d(4),
            nn.Conv2d(3, 32, 9, 1),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
        )
        
        # Residual blocks
        self.residuals = nn.Sequential(
            *[ResidualBlock(128) for _ in range(5)]
        )
        
        # Upsampling
        self.up = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            nn.ReflectionPad2d(4),
            nn.Conv2d(32, 3, 9, 1),
            nn.Tanh(),
        )
    
    def forward(self, x):
        x = self.down(x)
        x = self.residuals(x)
        x = self.up(x)
        return x


def style_loss(generated_features, style_features):
    """Compute style loss using Gram matrices."""
    def gram_matrix(x):
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)
    
    loss = 0
    for gen_feat, style_feat in zip(generated_features, style_features):
        loss += nn.functional.mse_loss(gram_matrix(gen_feat), gram_matrix(style_feat))
    return loss
```

## Face Generation and Editing

### Face Attribute Editing

```python
class FaceAttributeEditor(nn.Module):
    """Edit face attributes with attribute conditioning."""
    
    def __init__(self, num_attributes=40):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )
        
        self.attr_embed = nn.Linear(num_attributes, 512)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512 + 512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh(),
        )
    
    def forward(self, image, target_attributes):
        features = self.encoder(image)
        attr_emb = self.attr_embed(target_attributes)
        attr_emb = attr_emb.view(attr_emb.size(0), -1, 1, 1)
        attr_emb = attr_emb.expand(-1, -1, features.size(2), features.size(3))
        combined = torch.cat([features, attr_emb], dim=1)
        return self.decoder(combined)
```

## Data Augmentation

### GAN-Based Augmentation

```python
class DataAugmentationGAN:
    """Use GAN for intelligent data augmentation."""
    
    def __init__(self, generator, latent_dim=100, device='cpu'):
        self.generator = generator.to(device)
        self.latent_dim = latent_dim
        self.device = device
    
    def augment_batch(self, images, labels, num_augmentations=2):
        """Augment batch using GAN interpolation."""
        augmented_images = [images]
        augmented_labels = [labels]
        
        batch_size = images.size(0)
        
        for _ in range(num_augmentations):
            z1 = torch.randn(batch_size, self.latent_dim, device=self.device)
            z2 = torch.randn(batch_size, self.latent_dim, device=self.device)
            
            alpha = torch.rand(batch_size, 1, device=self.device)
            z_interp = alpha * z1 + (1 - alpha) * z2
            
            with torch.no_grad():
                generated = self.generator(z_interp, labels)
            
            augmented_images.append(generated)
            augmented_labels.append(labels)
        
        return torch.cat(augmented_images), torch.cat(augmented_labels)
```

## Summary

| Application | Key Technique | Output |
|-------------|---------------|--------|
| **Super-Resolution** | Residual learning, perceptual loss | High-res images |
| **Inpainting** | Dilated convolutions, context encoding | Completed images |
| **Style Transfer** | Gram matrices, instance norm | Stylized images |
| **Face Editing** | Attribute conditioning | Modified faces |
| **Augmentation** | Latent interpolation | Training data |

GANs enable diverse image synthesis applications, from enhancing resolution to creative editing.

---

# Healthcare Applications of GANs

GANs are transforming healthcare through medical image synthesis, drug discovery, and clinical data augmentation. This section covers practical applications with emphasis on responsible deployment.

## Medical Image Synthesis

### CT/MRI Image Generation

```python
import torch
import torch.nn as nn

class MedicalImageGenerator(nn.Module):
    """
    Generator for medical imaging modalities.
    
    Produces realistic CT, MRI, or X-ray images for training.
    """
    
    def __init__(self, latent_dim=100, num_classes=10, ngf=64):
        super().__init__()
        
        # Condition on pathology class
        self.label_embed = nn.Embedding(num_classes, latent_dim)
        
        self.main = nn.Sequential(
            # Input: (latent_dim * 2) x 1 x 1
            nn.ConvTranspose2d(latent_dim * 2, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf, 1, 4, 2, 1, bias=False),
            nn.Tanh(),
        )
    
    def forward(self, z, labels):
        label_emb = self.label_embed(labels)
        x = torch.cat([z, label_emb], dim=1)
        x = x.view(x.size(0), -1, 1, 1)
        return self.main(x)


class MedicalImageDiscriminator(nn.Module):
    """
    Discriminator with auxiliary pathology classifier.
    """
    
    def __init__(self, num_classes=10, ndf=64):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, ndf, 4, 2, 1, bias=False),
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
        self.adv_head = nn.Conv2d(ndf * 8, 1, 4, 1, 0)
        
        # Pathology classification head
        self.class_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ndf * 8 * 4 * 4, num_classes)
        )
    
    def forward(self, x):
        features = self.features(x)
        adv_out = self.adv_head(features).view(-1, 1)
        class_out = self.class_head(features)
        return adv_out, class_out
```

### Cross-Modality Translation

```python
class ModalityTranslator(nn.Module):
    """
    Translate between imaging modalities (e.g., CT to MRI).
    
    Uses CycleGAN-style architecture for unpaired translation.
    """
    
    def __init__(self, in_channels=1, out_channels=1, ngf=64, num_residuals=9):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, ngf, 7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
            
            nn.Conv2d(ngf, ngf * 2, 3, 2, 1),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(True),
            
            nn.Conv2d(ngf * 2, ngf * 4, 3, 2, 1),
            nn.InstanceNorm2d(ngf * 4),
            nn.ReLU(True),
        )
        
        # Residual blocks
        self.residuals = nn.Sequential(
            *[ResidualBlock(ngf * 4) for _ in range(num_residuals)]
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, 1),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf * 2, ngf, 3, 2, 1, 1),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
            
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_channels, 7),
            nn.Tanh(),
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.residuals(x)
        x = self.decoder(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
        )
    
    def forward(self, x):
        return x + self.block(x)
```

## Drug Discovery

### Molecular Generation with GANs

```python
class MolecularGenerator(nn.Module):
    """
    Generate molecular structures represented as SMILES strings.
    
    Uses character-level generation with LSTM.
    """
    
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, latent_dim=100):
        super().__init__()
        
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, z, max_len=100, start_token=0):
        """
        Generate SMILES string from latent vector.
        
        Args:
            z: Latent vector (batch, latent_dim)
            max_len: Maximum sequence length
            start_token: Start token index
        """
        batch_size = z.size(0)
        device = z.device
        
        # Initialize hidden state from latent
        h0 = self.latent_to_hidden(z).unsqueeze(0)
        c0 = torch.zeros_like(h0)
        hidden = (h0, c0)
        
        # Start token
        tokens = torch.full((batch_size, 1), start_token, 
                           dtype=torch.long, device=device)
        
        outputs = []
        
        for _ in range(max_len - 1):
            embed = self.embedding(tokens[:, -1:])
            output, hidden = self.lstm(embed, hidden)
            logits = self.output(output)
            
            # Sample next token
            probs = torch.softmax(logits.squeeze(1), dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            tokens = torch.cat([tokens, next_token], dim=1)
            outputs.append(logits)
        
        return tokens, torch.cat(outputs, dim=1)


class PropertyPredictor(nn.Module):
    """
    Predict molecular properties from SMILES embedding.
    
    Used to guide generation toward desired properties.
    """
    
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_properties=5):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, 
                           bidirectional=True)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, num_properties),
        )
    
    def forward(self, tokens):
        embed = self.embedding(tokens)
        output, (hidden, _) = self.lstm(embed)
        
        # Concatenate forward and backward final states
        combined = torch.cat([hidden[0], hidden[1]], dim=1)
        
        return self.predictor(combined)
```

## Synthetic Health Records

### Electronic Health Record Generation

```python
class EHRGenerator(nn.Module):
    """
    Generate synthetic electronic health records.
    
    Handles mixed data types: continuous, categorical, and temporal.
    """
    
    def __init__(self, latent_dim=100, continuous_dim=50, 
                 categorical_dims=[10, 5, 20], temporal_steps=12):
        super().__init__()
        
        self.continuous_dim = continuous_dim
        self.categorical_dims = categorical_dims
        self.temporal_steps = temporal_steps
        
        total_categorical = sum(categorical_dims)
        output_dim = continuous_dim + total_categorical
        
        # Static features generator
        self.static_gen = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )
        
        # Temporal features generator (lab values over time)
        self.temporal_gen = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, temporal_steps * 10),
        )
    
    def forward(self, z):
        batch_size = z.size(0)
        
        # Generate static features
        static = self.static_gen(z)
        
        # Split into continuous and categorical
        continuous = torch.tanh(static[:, :self.continuous_dim])
        
        categorical = []
        idx = self.continuous_dim
        for dim in self.categorical_dims:
            cat_logits = static[:, idx:idx+dim]
            cat_probs = torch.softmax(cat_logits, dim=-1)
            categorical.append(cat_probs)
            idx += dim
        
        # Generate temporal features
        temporal = self.temporal_gen(z)
        temporal = temporal.view(batch_size, self.temporal_steps, -1)
        
        return {
            'continuous': continuous,
            'categorical': categorical,
            'temporal': temporal,
        }


class EHRDiscriminator(nn.Module):
    """Discriminator for EHR data."""
    
    def __init__(self, continuous_dim=50, categorical_dims=[10, 5, 20], 
                 temporal_steps=12, temporal_features=10):
        super().__init__()
        
        total_categorical = sum(categorical_dims)
        static_dim = continuous_dim + total_categorical
        
        # Static feature processor
        self.static_net = nn.Sequential(
            nn.Linear(static_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
        )
        
        # Temporal feature processor
        self.temporal_net = nn.LSTM(temporal_features, 64, batch_first=True)
        
        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(128 + 64, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
        )
    
    def forward(self, continuous, categorical, temporal):
        # Combine static features
        cat_combined = torch.cat(categorical, dim=-1)
        static = torch.cat([continuous, cat_combined], dim=-1)
        
        static_feat = self.static_net(static)
        
        # Process temporal
        _, (temporal_hidden, _) = self.temporal_net(temporal)
        temporal_feat = temporal_hidden.squeeze(0)
        
        # Combine and classify
        combined = torch.cat([static_feat, temporal_feat], dim=-1)
        
        return self.classifier(combined)
```

## Privacy Considerations

### Differential Privacy for Medical GANs

```python
class DPMedicalGAN:
    """
    Medical GAN with differential privacy guarantees.
    """
    
    def __init__(self, generator, discriminator, epsilon=1.0, delta=1e-5,
                 max_grad_norm=1.0, noise_multiplier=1.0):
        self.G = generator
        self.D = discriminator
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
    
    def clip_gradients(self, model):
        """Clip per-sample gradients to bound sensitivity."""
        total_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        
        total_norm = total_norm ** 0.5
        clip_coef = self.max_grad_norm / (total_norm + 1e-6)
        
        if clip_coef < 1:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(clip_coef)
    
    def add_noise(self, model):
        """Add calibrated Gaussian noise for DP."""
        for param in model.parameters():
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * self.noise_multiplier * self.max_grad_norm
                param.grad.data.add_(noise)
    
    def train_step(self, real_data, optimizer_G, optimizer_D, criterion):
        batch_size = real_data.size(0)
        device = real_data.device
        
        # Train Discriminator with DP
        optimizer_D.zero_grad()
        
        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)
        
        output_real, _ = self.D(real_data)
        loss_real = criterion(output_real, real_labels)
        
        z = torch.randn(batch_size, 100, device=device)
        fake_data = self.G(z, torch.randint(0, 10, (batch_size,), device=device))
        output_fake, _ = self.D(fake_data.detach())
        loss_fake = criterion(output_fake, fake_labels)
        
        loss_D = loss_real + loss_fake
        loss_D.backward()
        
        # Apply DP: clip gradients and add noise
        self.clip_gradients(self.D)
        self.add_noise(self.D)
        
        optimizer_D.step()
        
        # Train Generator (no DP needed for generator)
        optimizer_G.zero_grad()
        
        output_fake, _ = self.D(fake_data)
        loss_G = criterion(output_fake, real_labels)
        loss_G.backward()
        optimizer_G.step()
        
        return loss_D.item(), loss_G.item()
```

## Summary

| Application | Data Type | Key Challenge |
|-------------|-----------|---------------|
| **Medical Imaging** | CT, MRI, X-ray | Anatomical accuracy |
| **Modality Translation** | Cross-modal | Preserving pathology |
| **Drug Discovery** | Molecular graphs | Chemical validity |
| **EHR Synthesis** | Mixed tabular | Temporal consistency |

Healthcare GANs require careful attention to privacy, validity, and clinical utility. Synthetic data should be validated by domain experts before deployment.

---

# Scientific Research Applications of GANs

GANs are accelerating scientific discovery across physics, astronomy, climate science, and materials research. This section covers applications where GANs simulate complex physical phenomena.

## Particle Physics

### Event Generation for High-Energy Physics

```python
import torch
import torch.nn as nn

class ParticleGenerator(nn.Module):
    """
    Generate particle collision events for physics simulation.
    
    Output: Particle 4-momenta (E, px, py, pz) for multiple particles.
    """
    
    def __init__(self, latent_dim=100, num_particles=10, condition_dim=5):
        super().__init__()
        
        self.num_particles = num_particles
        
        # Condition on collision parameters (energy, beam type, etc.)
        self.condition_embed = nn.Linear(condition_dim, latent_dim)
        
        self.main = nn.Sequential(
            nn.Linear(latent_dim * 2, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            
            nn.Linear(512, num_particles * 4),  # 4-momentum per particle
        )
    
    def forward(self, z, conditions):
        """
        Args:
            z: Latent noise (batch, latent_dim)
            conditions: Collision parameters (batch, condition_dim)
        """
        cond_emb = self.condition_embed(conditions)
        x = torch.cat([z, cond_emb], dim=1)
        
        output = self.main(x)
        
        # Reshape to (batch, num_particles, 4)
        momenta = output.view(-1, self.num_particles, 4)
        
        # Apply physical constraints
        momenta = self.apply_physics(momenta)
        
        return momenta
    
    def apply_physics(self, momenta):
        """
        Apply physical constraints to generated momenta.
        
        Ensures:
        - E >= |p| (mass shell condition)
        - Conservation laws approximately satisfied
        """
        # Ensure energy is positive and sufficient
        E = momenta[:, :, 0:1]
        p = momenta[:, :, 1:4]
        
        p_mag = torch.norm(p, dim=2, keepdim=True)
        E_min = p_mag + 0.001  # Small positive mass
        E_constrained = torch.maximum(E.abs(), E_min)
        
        return torch.cat([E_constrained, p], dim=2)


class ParticleDiscriminator(nn.Module):
    """Discriminator for particle physics events."""
    
    def __init__(self, num_particles=10, condition_dim=5):
        super().__init__()
        
        input_dim = num_particles * 4 + condition_dim
        
        self.main = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            
            nn.Linear(128, 1),
        )
    
    def forward(self, momenta, conditions):
        momenta_flat = momenta.view(momenta.size(0), -1)
        x = torch.cat([momenta_flat, conditions], dim=1)
        return self.main(x)


def physics_loss(momenta):
    """
    Additional loss for physical consistency.
    
    Penalizes:
    - Energy-momentum conservation violations
    - Mass shell violations
    """
    # Total momentum should be zero in CM frame
    total_momentum = momenta[:, :, 1:4].sum(dim=1)  # Sum over particles
    momentum_conservation = (total_momentum ** 2).sum(dim=1).mean()
    
    # Mass shell: E^2 - p^2 >= 0
    E = momenta[:, :, 0]
    p = momenta[:, :, 1:4]
    p_squared = (p ** 2).sum(dim=2)
    mass_squared = E ** 2 - p_squared
    mass_shell_violation = torch.relu(-mass_squared).mean()
    
    return momentum_conservation + mass_shell_violation
```

## Astronomy and Cosmology

### Galaxy Image Generation

```python
class GalaxyGenerator(nn.Module):
    """
    Generate realistic galaxy images.
    
    Conditioned on morphological type and redshift.
    """
    
    def __init__(self, latent_dim=100, num_types=5, ngf=64):
        super().__init__()
        
        # Morphology embedding
        self.morph_embed = nn.Embedding(num_types, latent_dim // 2)
        
        # Redshift embedding
        self.z_embed = nn.Linear(1, latent_dim // 2)
        
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim * 2, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Sigmoid(),  # Flux values in [0, 1]
        )
    
    def forward(self, z, morphology, redshift):
        """
        Args:
            z: Latent noise
            morphology: Galaxy type (integer)
            redshift: Cosmological redshift (float)
        """
        morph_emb = self.morph_embed(morphology)
        z_emb = self.z_embed(redshift.unsqueeze(-1))
        
        combined = torch.cat([z, morph_emb, z_emb], dim=1)
        combined = combined.view(combined.size(0), -1, 1, 1)
        
        return self.main(combined)


class CosmicWebGenerator(nn.Module):
    """
    Generate 3D cosmic web density fields.
    
    Models large-scale structure of the universe.
    """
    
    def __init__(self, latent_dim=100, ngf=32, output_size=64):
        super().__init__()
        
        self.main = nn.Sequential(
            # 3D transposed convolutions
            nn.ConvTranspose3d(latent_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm3d(ngf * 8),
            nn.ReLU(True),
            
            nn.ConvTranspose3d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ngf * 4),
            nn.ReLU(True),
            
            nn.ConvTranspose3d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ngf * 2),
            nn.ReLU(True),
            
            nn.ConvTranspose3d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ngf),
            nn.ReLU(True),
            
            nn.ConvTranspose3d(ngf, 1, 4, 2, 1, bias=False),
            nn.Softplus(),  # Density must be positive
        )
    
    def forward(self, z):
        z = z.view(z.size(0), -1, 1, 1, 1)
        return self.main(z)
```

## Climate Science

### Weather Pattern Generation

```python
class ClimateGenerator(nn.Module):
    """
    Generate realistic climate patterns.
    
    Outputs: Temperature, precipitation, pressure fields.
    """
    
    def __init__(self, latent_dim=100, num_channels=3, ngf=64):
        super().__init__()
        
        # Condition on time of year, ENSO state, etc.
        self.condition_net = nn.Sequential(
            nn.Linear(10, latent_dim),
            nn.ReLU(),
        )
        
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim * 2, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf, num_channels, 4, 2, 1, bias=False),
        )
        
        # Channel-specific activations
        self.temp_activation = nn.Identity()  # Temperature can be any value
        self.precip_activation = nn.Softplus()  # Precipitation must be positive
        self.pressure_activation = nn.Sigmoid()  # Pressure normalized to [0, 1]
    
    def forward(self, z, conditions):
        cond_emb = self.condition_net(conditions)
        combined = torch.cat([z, cond_emb], dim=1)
        combined = combined.view(combined.size(0), -1, 1, 1)
        
        output = self.main(combined)
        
        # Apply channel-specific activations
        temp = self.temp_activation(output[:, 0:1])
        precip = self.precip_activation(output[:, 1:2])
        pressure = self.pressure_activation(output[:, 2:3])
        
        return torch.cat([temp, precip, pressure], dim=1)


def climate_consistency_loss(generated, physical_constraints):
    """
    Enforce physical consistency in climate fields.
    
    - Spatial smoothness
    - Conservation laws
    - Physical bounds
    """
    # Spatial smoothness (penalize high-frequency noise)
    dx = generated[:, :, :, 1:] - generated[:, :, :, :-1]
    dy = generated[:, :, 1:, :] - generated[:, :, :-1, :]
    smoothness = (dx ** 2).mean() + (dy ** 2).mean()
    
    # Precipitation must be non-negative
    precip = generated[:, 1:2]
    precip_violation = torch.relu(-precip).mean()
    
    return smoothness + precip_violation
```

## Materials Science

### Crystal Structure Generation

```python
class CrystalGenerator(nn.Module):
    """
    Generate crystal structures with desired properties.
    
    Output: Atom positions and types in a unit cell.
    """
    
    def __init__(self, latent_dim=100, max_atoms=20, num_elements=100):
        super().__init__()
        
        self.max_atoms = max_atoms
        
        # Condition on desired properties (band gap, hardness, etc.)
        self.property_embed = nn.Linear(5, latent_dim)
        
        # Generate atom positions (x, y, z in [0, 1] fractional coords)
        self.position_net = nn.Sequential(
            nn.Linear(latent_dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, max_atoms * 3),
            nn.Sigmoid(),  # Fractional coordinates in [0, 1]
        )
        
        # Generate atom types
        self.type_net = nn.Sequential(
            nn.Linear(latent_dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, max_atoms * num_elements),
        )
        
        # Generate lattice parameters (a, b, c, alpha, beta, gamma)
        self.lattice_net = nn.Sequential(
            nn.Linear(latent_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 6),
            nn.Softplus(),  # Lattice params must be positive
        )
    
    def forward(self, z, target_properties):
        prop_emb = self.property_embed(target_properties)
        combined = torch.cat([z, prop_emb], dim=1)
        
        positions = self.position_net(combined)
        positions = positions.view(-1, self.max_atoms, 3)
        
        type_logits = self.type_net(combined)
        type_logits = type_logits.view(-1, self.max_atoms, -1)
        
        lattice = self.lattice_net(combined)
        
        return {
            'positions': positions,
            'type_logits': type_logits,
            'lattice': lattice,
        }


def crystal_validity_loss(crystal_data):
    """
    Penalize physically invalid crystal structures.
    """
    positions = crystal_data['positions']
    
    # Minimum distance constraint (atoms shouldn't overlap)
    batch_size = positions.size(0)
    n_atoms = positions.size(1)
    
    # Compute pairwise distances
    pos_expanded = positions.unsqueeze(2)  # (B, N, 1, 3)
    pos_tiled = positions.unsqueeze(1)  # (B, 1, N, 3)
    distances = torch.norm(pos_expanded - pos_tiled, dim=-1)  # (B, N, N)
    
    # Mask diagonal (self-distance)
    mask = 1 - torch.eye(n_atoms, device=positions.device)
    distances = distances * mask
    
    # Penalize distances below threshold
    min_distance = 0.1
    overlap_penalty = torch.relu(min_distance - distances).sum() / batch_size
    
    return overlap_penalty
```

## Summary

| Domain | Data Type | Physical Constraints |
|--------|-----------|---------------------|
| **Particle Physics** | 4-momenta | Energy-momentum conservation |
| **Astronomy** | Images, 3D fields | Morphology, redshift |
| **Climate** | Spatiotemporal | Conservation laws, bounds |
| **Materials** | Crystal structures | Minimum distances, symmetry |

Scientific GANs must incorporate domain-specific physical constraints to generate meaningful data. This often requires custom loss functions and architectures that respect fundamental physics.
