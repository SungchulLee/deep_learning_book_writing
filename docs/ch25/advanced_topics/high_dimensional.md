# High-Dimensional Data Synthesis with GANs

GANs extend beyond 2D images to generate complex, high-dimensional data including audio, video, 3D shapes, and text. This section covers architectures and techniques for these challenging domains.

## Audio Generation

### WaveGAN

WaveGAN generates raw audio waveforms using 1D transposed convolutions:

```python
import torch
import torch.nn as nn

class WaveGANGenerator(nn.Module):
    """
    Generator for raw audio waveform synthesis.
    
    Generates 16384 samples (about 1 second at 16kHz).
    """
    
    def __init__(self, latent_dim=100, model_size=64):
        super().__init__()
        
        d = model_size
        
        self.main = nn.Sequential(
            # Input: (batch, latent_dim, 1)
            nn.ConvTranspose1d(latent_dim, d * 32, 16, 1, 0),
            nn.BatchNorm1d(d * 32),
            nn.ReLU(True),
            # Size: 16
            
            nn.ConvTranspose1d(d * 32, d * 16, 25, 4, 11),
            nn.BatchNorm1d(d * 16),
            nn.ReLU(True),
            # Size: 64
            
            nn.ConvTranspose1d(d * 16, d * 8, 25, 4, 11),
            nn.BatchNorm1d(d * 8),
            nn.ReLU(True),
            # Size: 256
            
            nn.ConvTranspose1d(d * 8, d * 4, 25, 4, 11),
            nn.BatchNorm1d(d * 4),
            nn.ReLU(True),
            # Size: 1024
            
            nn.ConvTranspose1d(d * 4, d * 2, 25, 4, 11),
            nn.BatchNorm1d(d * 2),
            nn.ReLU(True),
            # Size: 4096
            
            nn.ConvTranspose1d(d * 2, 1, 25, 4, 11),
            nn.Tanh(),
            # Size: 16384
        )
    
    def forward(self, z):
        z = z.view(z.size(0), -1, 1)
        return self.main(z)


class WaveGANDiscriminator(nn.Module):
    """Discriminator for raw audio waveforms."""
    
    def __init__(self, model_size=64):
        super().__init__()
        
        d = model_size
        
        self.main = nn.Sequential(
            # Input: (batch, 1, 16384)
            nn.Conv1d(1, d, 25, 4, 11),
            nn.LeakyReLU(0.2),
            # Size: 4096
            
            nn.Conv1d(d, d * 2, 25, 4, 11),
            nn.BatchNorm1d(d * 2),
            nn.LeakyReLU(0.2),
            # Size: 1024
            
            nn.Conv1d(d * 2, d * 4, 25, 4, 11),
            nn.BatchNorm1d(d * 4),
            nn.LeakyReLU(0.2),
            # Size: 256
            
            nn.Conv1d(d * 4, d * 8, 25, 4, 11),
            nn.BatchNorm1d(d * 8),
            nn.LeakyReLU(0.2),
            # Size: 64
            
            nn.Conv1d(d * 8, d * 16, 25, 4, 11),
            nn.BatchNorm1d(d * 16),
            nn.LeakyReLU(0.2),
            # Size: 16
            
            nn.Conv1d(d * 16, 1, 16, 1, 0),
            # Size: 1
        )
    
    def forward(self, x):
        return self.main(x).view(-1, 1)
```

### Spectrogram-Based Audio GAN

```python
class MelSpecGAN(nn.Module):
    """
    Generate mel spectrograms and convert to audio using vocoder.
    """
    
    def __init__(self, latent_dim=100, mel_channels=80, time_steps=128):
        super().__init__()
        
        self.time_steps = time_steps
        self.mel_channels = mel_channels
        
        # Generator produces mel spectrogram
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, 256 * 8 * 5),
            nn.ReLU(True),
            nn.Unflatten(1, (256, 8, 5)),
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 16 x 10
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 32 x 20
            
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # 64 x 40
            
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            # 128 x 80 -> mel_channels x time_steps
        )
    
    def forward(self, z):
        mel = self.generator(z)
        return mel.squeeze(1).transpose(1, 2)  # (batch, time, mel_channels)
```

## Video Generation

### Video GAN with Temporal Coherence

```python
class VideoGenerator(nn.Module):
    """
    Generate video sequences with temporal consistency.
    
    Uses 3D convolutions to capture spatial and temporal patterns.
    """
    
    def __init__(self, latent_dim=100, num_frames=16, image_size=64):
        super().__init__()
        
        self.num_frames = num_frames
        
        # Temporal latent transformation
        self.fc = nn.Linear(latent_dim, 512 * 4 * 4 * 4)
        
        # 3D transposed convolutions
        self.main = nn.Sequential(
            # Input: 512 x 4 x 4 x 4 (C x T x H x W)
            nn.ConvTranspose3d(512, 256, (4, 4, 4), (2, 2, 2), (1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(True),
            # 256 x 8 x 8 x 8
            
            nn.ConvTranspose3d(256, 128, (4, 4, 4), (2, 2, 2), (1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(True),
            # 128 x 16 x 16 x 16
            
            nn.ConvTranspose3d(128, 64, (4, 4, 4), (1, 2, 2), (0, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            # 64 x 16 x 32 x 32
            
            nn.ConvTranspose3d(64, 3, (4, 4, 4), (1, 2, 2), (0, 1, 1)),
            nn.Tanh(),
            # 3 x 16 x 64 x 64
        )
    
    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 512, 4, 4, 4)
        video = self.main(x)
        return video  # (batch, channels, frames, height, width)


class VideoDiscriminator(nn.Module):
    """Discriminator for video sequences."""
    
    def __init__(self, num_frames=16, image_size=64):
        super().__init__()
        
        self.main = nn.Sequential(
            # Input: 3 x 16 x 64 x 64
            nn.Conv3d(3, 64, (4, 4, 4), (1, 2, 2), (0, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv3d(64, 128, (4, 4, 4), (1, 2, 2), (0, 1, 1)),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv3d(128, 256, (4, 4, 4), (2, 2, 2), (1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv3d(256, 512, (4, 4, 4), (2, 2, 2), (1, 1, 1)),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv3d(512, 1, (4, 4, 4), (1, 1, 1), (0, 0, 0)),
        )
    
    def forward(self, video):
        return self.main(video).view(-1, 1)


class TemporalDiscriminator(nn.Module):
    """
    Discriminator focusing on temporal coherence.
    
    Uses frame differences to detect temporal artifacts.
    """
    
    def __init__(self, image_size=64):
        super().__init__()
        
        # Process frame differences
        self.spatial_net = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )
        
        # Temporal aggregation
        self.temporal_net = nn.LSTM(256 * 8 * 8, 512, batch_first=True)
        
        self.classifier = nn.Linear(512, 1)
    
    def forward(self, video):
        batch_size = video.size(0)
        num_frames = video.size(2)
        
        # Process each frame
        frame_features = []
        for t in range(num_frames):
            frame = video[:, :, t, :, :]
            feat = self.spatial_net(frame)
            frame_features.append(feat.view(batch_size, -1))
        
        frame_features = torch.stack(frame_features, dim=1)
        
        # Temporal processing
        _, (hidden, _) = self.temporal_net(frame_features)
        
        return self.classifier(hidden.squeeze(0))
```

## 3D Shape Generation

### 3D-GAN for Voxel Generation

```python
class VoxelGenerator(nn.Module):
    """
    Generate 3D voxel grids.
    
    Output: 64 x 64 x 64 binary occupancy grid.
    """
    
    def __init__(self, latent_dim=200):
        super().__init__()
        
        self.main = nn.Sequential(
            # Input: latent_dim x 1 x 1 x 1
            nn.ConvTranspose3d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm3d(512),
            nn.ReLU(True),
            # 512 x 4 x 4 x 4
            
            nn.ConvTranspose3d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(True),
            # 256 x 8 x 8 x 8
            
            nn.ConvTranspose3d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(True),
            # 128 x 16 x 16 x 16
            
            nn.ConvTranspose3d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            # 64 x 32 x 32 x 32
            
            nn.ConvTranspose3d(64, 1, 4, 2, 1, bias=False),
            nn.Sigmoid(),
            # 1 x 64 x 64 x 64
        )
    
    def forward(self, z):
        z = z.view(z.size(0), -1, 1, 1, 1)
        return self.main(z)


class VoxelDiscriminator(nn.Module):
    """Discriminator for 3D voxel grids."""
    
    def __init__(self):
        super().__init__()
        
        self.main = nn.Sequential(
            nn.Conv3d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv3d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv3d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv3d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv3d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        return self.main(x).view(-1, 1)
```

### Point Cloud Generation

```python
class PointCloudGenerator(nn.Module):
    """
    Generate 3D point clouds.
    
    Output: N points with (x, y, z) coordinates.
    """
    
    def __init__(self, latent_dim=100, num_points=2048):
        super().__init__()
        
        self.num_points = num_points
        
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, num_points * 3),
            nn.Tanh(),  # Points in [-1, 1]^3
        )
    
    def forward(self, z):
        points = self.fc(z)
        return points.view(-1, self.num_points, 3)


class PointCloudDiscriminator(nn.Module):
    """
    Discriminator for point clouds using PointNet architecture.
    """
    
    def __init__(self, num_points=2048):
        super().__init__()
        
        # Point-wise MLP
        self.point_net = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Conv1d(256, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
        )
        
        # Global feature aggregation (max pooling)
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, points):
        # points: (batch, num_points, 3)
        x = points.transpose(1, 2)  # (batch, 3, num_points)
        
        x = self.point_net(x)  # (batch, 1024, num_points)
        
        # Global max pooling
        x = x.max(dim=2)[0]  # (batch, 1024)
        
        return self.classifier(x)
```

## Text Generation with GANs

### SeqGAN for Discrete Sequences

```python
class TextGenerator(nn.Module):
    """
    LSTM-based text generator for SeqGAN.
    """
    
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden=None):
        """
        Forward pass for teacher forcing.
        
        Args:
            x: Input tokens (batch, seq_len)
            hidden: LSTM hidden state
        
        Returns:
            Logits for next token prediction
        """
        embed = self.embedding(x)
        output, hidden = self.lstm(embed, hidden)
        logits = self.output(output)
        return logits, hidden
    
    def sample(self, batch_size, max_len, start_token, device='cpu'):
        """
        Sample sequences autoregressively.
        """
        # Start with start token
        tokens = torch.full((batch_size, 1), start_token, 
                           dtype=torch.long, device=device)
        hidden = None
        
        for _ in range(max_len - 1):
            logits, hidden = self.forward(tokens[:, -1:], hidden)
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, 1)
            tokens = torch.cat([tokens, next_token], dim=1)
        
        return tokens


class TextDiscriminator(nn.Module):
    """
    CNN-based discriminator for text sequences.
    """
    
    def __init__(self, vocab_size, embed_dim=128, num_filters=128, 
                 filter_sizes=[2, 3, 4, 5]):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Multiple filter sizes for different n-gram patterns
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k)
            for k in filter_sizes
        ])
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_filters * len(filter_sizes), 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        """
        Args:
            x: Token indices (batch, seq_len)
        """
        embed = self.embedding(x)  # (batch, seq_len, embed_dim)
        embed = embed.transpose(1, 2)  # (batch, embed_dim, seq_len)
        
        # Apply convolutions with different filter sizes
        conv_outputs = []
        for conv in self.convs:
            h = torch.relu(conv(embed))  # (batch, num_filters, seq_len - k + 1)
            h = h.max(dim=2)[0]  # Global max pooling
            conv_outputs.append(h)
        
        # Concatenate all filter outputs
        features = torch.cat(conv_outputs, dim=1)
        
        return self.classifier(features)
```

## Summary

| Domain | Key Architecture | Challenges |
|--------|-----------------|------------|
| **Audio** | WaveGAN (1D conv), MelGAN | Long-range coherence |
| **Video** | 3D conv, temporal discriminator | Temporal consistency |
| **3D Shapes** | Voxel GAN, PointCloud GAN | Geometric accuracy |
| **Text** | SeqGAN, LSTM + RL | Discrete outputs |

High-dimensional synthesis requires domain-specific architectures that capture the unique structure of each data type while maintaining GAN training stability.

---

# Multi-Modal GANs

Multi-modal GANs generate or translate between different data modalities, enabling applications like text-to-image synthesis, image captioning, and cross-modal retrieval.

## Cross-Modal Generation

### Text-to-Image Synthesis

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TextEncoder(nn.Module):
    """
    Encode text descriptions into conditioning vectors.
    """
    
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=256, output_dim=256):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.projection = nn.Linear(hidden_dim * 2, output_dim)
    
    def forward(self, text, text_lengths):
        """
        Args:
            text: Token indices (batch, max_len)
            text_lengths: Actual lengths (batch,)
        
        Returns:
            text_embedding: (batch, output_dim)
            word_embeddings: (batch, max_len, hidden_dim * 2)
        """
        embed = self.embedding(text)
        
        # Pack for variable length
        packed = nn.utils.rnn.pack_padded_sequence(
            embed, text_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        outputs, (hidden, _) = self.lstm(packed)
        
        # Unpack
        word_embeddings, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        
        # Concatenate forward and backward final hidden states
        text_embedding = torch.cat([hidden[0], hidden[1]], dim=1)
        text_embedding = self.projection(text_embedding)
        
        return text_embedding, word_embeddings


class TextConditionedGenerator(nn.Module):
    """
    Generator conditioned on text embeddings.
    """
    
    def __init__(self, latent_dim=100, text_dim=256, ngf=64, image_channels=3):
        super().__init__()
        
        # Conditioning augmentation
        self.ca_net = nn.Sequential(
            nn.Linear(text_dim, text_dim * 2),
            nn.LeakyReLU(0.2),
        )
        
        # Combined input: noise + conditioned text
        input_dim = latent_dim + text_dim
        
        self.main = nn.Sequential(
            nn.ConvTranspose2d(input_dim, ngf * 8, 4, 1, 0, bias=False),
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
            
            nn.ConvTranspose2d(ngf, image_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )
    
    def conditioning_augmentation(self, text_embedding):
        """
        Add noise to text embedding for diversity.
        
        Outputs mean and log variance, samples from Gaussian.
        """
        x = self.ca_net(text_embedding)
        mu, logvar = x.chunk(2, dim=1)
        
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        conditioned = mu + eps * std
        
        return conditioned, mu, logvar
    
    def forward(self, z, text_embedding):
        conditioned, mu, logvar = self.conditioning_augmentation(text_embedding)
        
        # Concatenate noise and text condition
        combined = torch.cat([z, conditioned], dim=1)
        combined = combined.view(combined.size(0), -1, 1, 1)
        
        image = self.main(combined)
        
        return image, mu, logvar


class TextConditionedDiscriminator(nn.Module):
    """
    Discriminator that takes both image and text.
    """
    
    def __init__(self, text_dim=256, ndf=64, image_channels=3):
        super().__init__()
        
        # Image encoder
        self.image_encoder = nn.Sequential(
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
        
        # Text projection to match image feature map
        self.text_proj = nn.Linear(text_dim, ndf * 8)
        
        # Joint classifier
        self.classifier = nn.Sequential(
            nn.Conv2d(ndf * 8 + ndf * 8, ndf * 8, 1, 1, 0),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0),
        )
    
    def forward(self, image, text_embedding):
        # Encode image
        img_features = self.image_encoder(image)  # (batch, ndf*8, 4, 4)
        
        # Project and replicate text
        text_proj = self.text_proj(text_embedding)  # (batch, ndf*8)
        text_proj = text_proj.view(text_proj.size(0), -1, 1, 1)
        text_proj = text_proj.repeat(1, 1, 4, 4)  # (batch, ndf*8, 4, 4)
        
        # Concatenate and classify
        combined = torch.cat([img_features, text_proj], dim=1)
        output = self.classifier(combined)
        
        return output.view(-1, 1)
```

### Text-to-Image Training

```python
def train_text_to_image_gan(G, D, text_encoder, dataloader, num_epochs,
                             latent_dim=100, device='cpu'):
    """
    Train text-to-image GAN.
    """
    G.to(device)
    D.to(device)
    text_encoder.to(device)
    
    optimizer_G = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(num_epochs):
        for images, captions, lengths in dataloader:
            batch_size = images.size(0)
            images = images.to(device)
            captions = captions.to(device)
            
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)
            
            # Encode text
            with torch.no_grad():
                text_emb, _ = text_encoder(captions, lengths)
            
            # =================================================================
            # Train Discriminator
            # =================================================================
            optimizer_D.zero_grad()
            
            # Real images with matching text
            output_real = D(images, text_emb)
            loss_real = criterion(output_real, real_labels)
            
            # Fake images with matching text
            z = torch.randn(batch_size, latent_dim, device=device)
            fake_images, _, _ = G(z, text_emb)
            output_fake = D(fake_images.detach(), text_emb)
            loss_fake = criterion(output_fake, fake_labels)
            
            # Mismatched: real images with wrong text
            wrong_text = text_emb[torch.randperm(batch_size)]
            output_wrong = D(images, wrong_text)
            loss_wrong = criterion(output_wrong, fake_labels)
            
            loss_D = loss_real + 0.5 * (loss_fake + loss_wrong)
            loss_D.backward()
            optimizer_D.step()
            
            # =================================================================
            # Train Generator
            # =================================================================
            optimizer_G.zero_grad()
            
            fake_images, mu, logvar = G(z, text_emb)
            output = D(fake_images, text_emb)
            
            # Adversarial loss
            loss_G_adv = criterion(output, real_labels)
            
            # KL divergence for conditioning augmentation
            loss_KL = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            
            loss_G = loss_G_adv + 2.0 * loss_KL
            loss_G.backward()
            optimizer_G.step()
        
        print(f'Epoch {epoch+1}/{num_epochs}: D_loss={loss_D.item():.4f}, G_loss={loss_G.item():.4f}')
```

## Image-to-Image Translation

### Multi-Domain Translation with StarGAN

```python
class StarGANGenerator(nn.Module):
    """
    Generator for multi-domain image-to-image translation.
    
    Takes image and target domain label, outputs translated image.
    """
    
    def __init__(self, image_channels=3, num_domains=5, ngf=64, num_residuals=6):
        super().__init__()
        
        # Down-sampling
        self.down = nn.Sequential(
            nn.Conv2d(image_channels + num_domains, ngf, 7, 1, 3, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ngf * 4),
            nn.ReLU(inplace=True),
        )
        
        # Residual blocks
        self.residuals = nn.Sequential(
            *[ResidualBlock(ngf * 4) for _ in range(num_residuals)]
        )
        
        # Up-sampling
        self.up = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(ngf, image_channels, 7, 1, 3),
            nn.Tanh(),
        )
    
    def forward(self, x, target_domain):
        """
        Args:
            x: Input image (batch, C, H, W)
            target_domain: One-hot domain label (batch, num_domains)
        """
        # Replicate domain label across spatial dimensions
        domain_map = target_domain.view(target_domain.size(0), -1, 1, 1)
        domain_map = domain_map.repeat(1, 1, x.size(2), x.size(3))
        
        # Concatenate image and domain
        x = torch.cat([x, domain_map], dim=1)
        
        x = self.down(x)
        x = self.residuals(x)
        x = self.up(x)
        
        return x


class ResidualBlock(nn.Module):
    """Residual block with instance normalization."""
    
    def __init__(self, channels):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(channels),
        )
    
    def forward(self, x):
        return x + self.block(x)


class StarGANDiscriminator(nn.Module):
    """
    Multi-task discriminator for StarGAN.
    
    Outputs: real/fake score AND domain classification.
    """
    
    def __init__(self, image_size=128, image_channels=3, num_domains=5, ndf=64):
        super().__init__()
        
        # Calculate feature map size after downsampling
        self.output_size = image_size // 32
        
        self.main = nn.Sequential(
            nn.Conv2d(image_channels, ndf, 4, 2, 1),
            nn.LeakyReLU(0.01, inplace=True),
            
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1),
            nn.LeakyReLU(0.01, inplace=True),
            
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1),
            nn.LeakyReLU(0.01, inplace=True),
            
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1),
            nn.LeakyReLU(0.01, inplace=True),
            
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1),
            nn.LeakyReLU(0.01, inplace=True),
        )
        
        # Real/Fake output
        self.src_conv = nn.Conv2d(ndf * 16, 1, 3, 1, 1)
        
        # Domain classification output
        self.cls_conv = nn.Conv2d(ndf * 16, num_domains, self.output_size, 1, 0)
    
    def forward(self, x):
        features = self.main(x)
        
        src_out = self.src_conv(features)  # (batch, 1, H', W')
        cls_out = self.cls_conv(features)  # (batch, num_domains, 1, 1)
        
        return src_out, cls_out.view(cls_out.size(0), -1)
```

## Audio-Visual Synthesis

### Sound-to-Image Generation

```python
class AudioVisualGenerator(nn.Module):
    """
    Generate images from audio spectrograms.
    """
    
    def __init__(self, audio_channels=1, ngf=64):
        super().__init__()
        
        # Audio encoder
        self.audio_encoder = nn.Sequential(
            nn.Conv2d(audio_channels, 64, 4, 2, 1),
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
            nn.AdaptiveAvgPool2d(1),
        )
        
        # Image decoder
        self.image_decoder = nn.Sequential(
            nn.ConvTranspose2d(512, ngf * 8, 4, 1, 0, bias=False),
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
            nn.Tanh(),
        )
    
    def forward(self, audio_spectrogram):
        """
        Args:
            audio_spectrogram: (batch, 1, freq_bins, time_frames)
        
        Returns:
            Generated image: (batch, 3, H, W)
        """
        audio_features = self.audio_encoder(audio_spectrogram)
        image = self.image_decoder(audio_features)
        return image
```

## Joint Embedding Space

### CLIP-Guided Generation

```python
class CLIPGuidedGenerator(nn.Module):
    """
    Generator guided by CLIP embeddings for better text-image alignment.
    """
    
    def __init__(self, clip_dim=512, latent_dim=512, ngf=64):
        super().__init__()
        
        # Map CLIP embedding to generator input
        self.clip_projection = nn.Sequential(
            nn.Linear(clip_dim, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, latent_dim),
        )
        
        # Standard generator architecture
        self.generator = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, ngf * 8, 4, 1, 0, bias=False),
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
            nn.Tanh(),
        )
    
    def forward(self, clip_embedding, noise=None):
        """
        Generate image from CLIP text embedding.
        
        Args:
            clip_embedding: CLIP text embedding (batch, clip_dim)
            noise: Optional noise for diversity
        """
        latent = self.clip_projection(clip_embedding)
        
        if noise is not None:
            latent = latent + 0.1 * noise
        
        latent = latent.view(latent.size(0), -1, 1, 1)
        image = self.generator(latent)
        
        return image


def clip_guided_loss(generated_images, text_embeddings, clip_model):
    """
    Compute CLIP-guided loss for better text-image alignment.
    
    Args:
        generated_images: Generated images (batch, 3, H, W)
        text_embeddings: CLIP text embeddings (batch, clip_dim)
        clip_model: Pre-trained CLIP model
    
    Returns:
        Cosine similarity loss
    """
    # Get CLIP image embeddings
    image_embeddings = clip_model.encode_image(generated_images)
    
    # Normalize embeddings
    image_embeddings = F.normalize(image_embeddings, dim=-1)
    text_embeddings = F.normalize(text_embeddings, dim=-1)
    
    # Cosine similarity (higher is better)
    similarity = (image_embeddings * text_embeddings).sum(dim=-1)
    
    # Loss: negative similarity
    loss = -similarity.mean()
    
    return loss
```

## Summary

| Task | Input | Output | Key Challenge |
|------|-------|--------|---------------|
| **Text-to-Image** | Text description | Image | Semantic alignment |
| **Image Captioning** | Image | Text | Language generation |
| **Multi-Domain** | Image + domain | Translated image | Preserving content |
| **Audio-Visual** | Audio | Video/Image | Cross-modal mapping |
| **CLIP-Guided** | Text embedding | Image | Fine-grained control |

Multi-modal GANs bridge different data types, enabling rich cross-modal applications. Key techniques include joint embedding spaces, conditioning mechanisms, and cycle consistency for unpaired translation.
