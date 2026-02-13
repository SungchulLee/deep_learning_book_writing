"""
Practical Examples and Applications
====================================

Real-world examples showing when and how to use different normalization layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Example 1: ResNet-style Block with BatchNorm (Image Classification)
# ============================================================================

class ResNetBlock(nn.Module):
    """
    Standard ResNet block using Batch Normalization.
    Best for image classification with moderate-to-large batch sizes.
    """
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ImageClassifier(nn.Module):
    """
    Complete image classification network with BatchNorm.
    """
    
    def __init__(self, num_classes=10):
        super(ImageClassifier, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(ResNetBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResNetBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


# ============================================================================
# Example 2: Transformer Block with LayerNorm (NLP/Transformers)
# ============================================================================

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.
    """
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections and reshape for multi-head
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        output = torch.matmul(attn, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(output)
        
        return output


class TransformerBlock(nn.Module):
    """
    Transformer block using Layer Normalization.
    Standard architecture for NLP and sequence modeling.
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual connection and layer norm
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class TransformerEncoder(nn.Module):
    """
    Complete Transformer encoder with LayerNorm.
    Used in BERT, GPT, and other transformer models.
    """
    
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6, 
                 d_ff=2048, max_len=5000, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = self._create_positional_encoding(max_len, d_model)
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def _create_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)
    
    def forward(self, x, mask=None):
        seq_len = x.size(1)
        
        # Embedding and positional encoding
        x = self.embedding(x) * (self.embedding.embedding_dim ** 0.5)
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)
        
        # Pass through transformer blocks
        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.norm(x)
        
        return x


# ============================================================================
# Example 3: U-Net with GroupNorm (Semantic Segmentation)
# ============================================================================

class DoubleConv(nn.Module):
    """
    Double convolution block with GroupNorm for U-Net.
    GroupNorm works well for segmentation with small batches.
    """
    
    def __init__(self, in_channels, out_channels, num_groups=8):
        super(DoubleConv, self).__init__()
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    """
    U-Net architecture with GroupNorm for semantic segmentation.
    GroupNorm is preferred over BatchNorm for small batch sizes.
    """
    
    def __init__(self, in_channels=3, num_classes=2, num_groups=8):
        super(UNet, self).__init__()
        
        # Encoder (downsampling)
        self.enc1 = DoubleConv(in_channels, 64, num_groups)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = DoubleConv(64, 128, num_groups)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = DoubleConv(128, 256, num_groups)
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = DoubleConv(256, 512, num_groups)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024, num_groups)
        
        # Decoder (upsampling)
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512, num_groups)
        
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256, num_groups)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128, num_groups)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64, num_groups)
        
        # Final output
        self.out = nn.Conv2d(64, num_classes, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        # Decoder with skip connections
        dec4 = self.up4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.up3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.up2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.up1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        return self.out(dec1)


# ============================================================================
# Example 4: CycleGAN Generator with InstanceNorm (Style Transfer)
# ============================================================================

class ResidualBlockInstanceNorm(nn.Module):
    """
    Residual block with Instance Normalization for image-to-image translation.
    """
    
    def __init__(self, channels):
        super(ResidualBlockInstanceNorm, self).__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
        )
    
    def forward(self, x):
        return x + self.block(x)


class CycleGANGenerator(nn.Module):
    """
    CycleGAN generator with Instance Normalization.
    Used for unpaired image-to-image translation.
    """
    
    def __init__(self, input_channels=3, output_channels=3, ngf=64, num_residual_blocks=9):
        super(CycleGANGenerator, self).__init__()
        
        # Initial convolution
        model = [
            nn.Conv2d(input_channels, ngf, kernel_size=7, padding=3, bias=False),
            nn.InstanceNorm2d(ngf, affine=True),
            nn.ReLU(inplace=True)
        ]
        
        # Downsampling
        for i in range(2):
            mult = 2 ** i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(ngf * mult * 2, affine=True),
                nn.ReLU(inplace=True)
            ]
        
        # Residual blocks
        mult = 2 ** 2
        for i in range(num_residual_blocks):
            model += [ResidualBlockInstanceNorm(ngf * mult)]
        
        # Upsampling
        for i in range(2):
            mult = 2 ** (2 - i)
            model += [
                nn.ConvTranspose2d(ngf * mult, ngf * mult // 2, 
                                  kernel_size=3, stride=2, padding=1, 
                                  output_padding=1, bias=False),
                nn.InstanceNorm2d(ngf * mult // 2, affine=True),
                nn.ReLU(inplace=True)
            ]
        
        # Output layer
        model += [
            nn.Conv2d(ngf, output_channels, kernel_size=7, padding=3),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)


# ============================================================================
# Example 5: LSTM with LayerNorm (Sequence Modeling)
# ============================================================================

class LayerNormLSTMCell(nn.Module):
    """
    LSTM cell with Layer Normalization.
    Improves training stability for deep RNNs.
    """
    
    def __init__(self, input_size, hidden_size):
        super(LayerNormLSTMCell, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Gates
        self.W_i = nn.Linear(input_size, hidden_size)
        self.U_i = nn.Linear(hidden_size, hidden_size, bias=False)
        self.ln_i = nn.LayerNorm(hidden_size)
        
        self.W_f = nn.Linear(input_size, hidden_size)
        self.U_f = nn.Linear(hidden_size, hidden_size, bias=False)
        self.ln_f = nn.LayerNorm(hidden_size)
        
        self.W_c = nn.Linear(input_size, hidden_size)
        self.U_c = nn.Linear(hidden_size, hidden_size, bias=False)
        self.ln_c = nn.LayerNorm(hidden_size)
        
        self.W_o = nn.Linear(input_size, hidden_size)
        self.U_o = nn.Linear(hidden_size, hidden_size, bias=False)
        self.ln_o = nn.LayerNorm(hidden_size)
    
    def forward(self, x, states):
        h, c = states
        
        # Input gate
        i = torch.sigmoid(self.ln_i(self.W_i(x) + self.U_i(h)))
        
        # Forget gate
        f = torch.sigmoid(self.ln_f(self.W_f(x) + self.U_f(h)))
        
        # Cell gate
        c_tilde = torch.tanh(self.ln_c(self.W_c(x) + self.U_c(h)))
        
        # Output gate
        o = torch.sigmoid(self.ln_o(self.W_o(x) + self.U_o(h)))
        
        # New cell state
        c_new = f * c + i * c_tilde
        
        # New hidden state
        h_new = o * torch.tanh(c_new)
        
        return h_new, c_new


class LayerNormLSTM(nn.Module):
    """
    Complete LSTM with Layer Normalization for sequence modeling.
    """
    
    def __init__(self, input_size, hidden_size, num_layers):
        super(LayerNormLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.cells = nn.ModuleList([
            LayerNormLSTMCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
    
    def forward(self, x, states=None):
        batch_size, seq_len, _ = x.size()
        
        if states is None:
            h = [torch.zeros(batch_size, self.hidden_size, device=x.device) 
                 for _ in range(self.num_layers)]
            c = [torch.zeros(batch_size, self.hidden_size, device=x.device) 
                 for _ in range(self.num_layers)]
        else:
            h, c = states
        
        outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            for layer in range(self.num_layers):
                h[layer], c[layer] = self.cells[layer](x_t, (h[layer], c[layer]))
                x_t = h[layer]
            
            outputs.append(h[-1].unsqueeze(1))
        
        outputs = torch.cat(outputs, dim=1)
        
        return outputs, (h, c)


# ============================================================================
# Usage Examples and Testing
# ============================================================================

def test_all_examples():
    """
    Test all example networks with sample inputs.
    """
    print("=" * 70)
    print("Testing All Example Networks")
    print("=" * 70)
    
    # Test Image Classifier with BatchNorm
    print("\n1. Image Classifier (BatchNorm):")
    classifier = ImageClassifier(num_classes=10)
    x = torch.randn(8, 3, 224, 224)
    out = classifier(x)
    print(f"   Input: {x.shape} → Output: {out.shape}")
    
    # Test Transformer with LayerNorm
    print("\n2. Transformer Encoder (LayerNorm):")
    transformer = TransformerEncoder(vocab_size=10000, d_model=512)
    x = torch.randint(0, 10000, (8, 50))  # batch=8, seq_len=50
    out = transformer(x)
    print(f"   Input: {x.shape} → Output: {out.shape}")
    
    # Test U-Net with GroupNorm
    print("\n3. U-Net Segmentation (GroupNorm):")
    unet = UNet(in_channels=3, num_classes=2)
    x = torch.randn(2, 3, 256, 256)  # Small batch
    out = unet(x)
    print(f"   Input: {x.shape} → Output: {out.shape}")
    
    # Test CycleGAN with InstanceNorm
    print("\n4. CycleGAN Generator (InstanceNorm):")
    generator = CycleGANGenerator()
    x = torch.randn(4, 3, 256, 256)
    out = generator(x)
    print(f"   Input: {x.shape} → Output: {out.shape}")
    
    # Test LSTM with LayerNorm
    print("\n5. LayerNorm LSTM:")
    lstm = LayerNormLSTM(input_size=100, hidden_size=256, num_layers=2)
    x = torch.randn(8, 20, 100)  # batch=8, seq_len=20, features=100
    out, _ = lstm(x)
    print(f"   Input: {x.shape} → Output: {out.shape}")
    
    print("\n" + "=" * 70)
    print("All tests passed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    test_all_examples()
    
    print("\n" + "=" * 70)
    print("Summary of Examples:")
    print("=" * 70)
    print("""
    1. Image Classification → BatchNorm
       - Standard choice for CNNs
       - Works well with moderate-to-large batches
    
    2. Transformers/NLP → LayerNorm
       - Standard for all transformer architectures
       - Batch-independent, stable training
    
    3. Semantic Segmentation → GroupNorm
       - Better than BatchNorm for small batches
       - Common in medical imaging and segmentation
    
    4. Style Transfer/GANs → InstanceNorm
       - Removes instance-specific information
       - Essential for image-to-image translation
    
    5. RNNs/LSTMs → LayerNorm
       - Improves training of deep RNNs
       - Handles variable sequence lengths well
    """)
