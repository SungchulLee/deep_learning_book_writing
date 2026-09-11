"""dcgan — dcgan 페이지의 코드를 모듈로 쓸 수 있게 옮겨 놓은 것이다.

이 파일은 appendix/generative/dcgan.md 에서 자동으로 뽑아낸 것이므로, 고칠 일이 있으면
그 페이지를 고치고 다시 뽑아내는 편이 낫다.
"""

#!/usr/bin/env python3
'''
DCGAN - 깊은 엮음 GAN
논문: "Unsupervised Representation Learning with Deep Convolutional GANs" (2015)
고갱이: 다층 퍼셉트론을 엮음 층으로 갈음하고 GAN 얼개의 길잡이를 준다
'''
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

class DCGenerator(nn.Module):
    def __init__(self, latent_dim=100, channels=1):
        super().__init__()
        self.init_size = 7
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )
    
    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class DCDiscriminator(nn.Module):
    def __init__(self, channels=1):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(channels, 16, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, 3, 1, 0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.adv_layer = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())
    
    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity

class DCGAN(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.generator = DCGenerator(latent_dim)
        self.discriminator = DCDiscriminator()
    
    def forward(self, z):
        return self.generator(z)

if __name__ == "__main__":
    model = DCGAN()
    print(f"Generator Parameters: {sum(p.numel() for p in model.generator.parameters()):,}")
    print(f"Discriminator Parameters: {sum(p.numel() for p in model.discriminator.parameters()):,}")
