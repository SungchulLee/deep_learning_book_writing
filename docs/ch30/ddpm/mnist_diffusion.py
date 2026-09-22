"""mnist_diffusion — mnist_diffusion 페이지의 코드를 모듈로 쓸 수 있게 옮겨 놓은 것이다.

이 파일은 ch28/ddpm/mnist_diffusion.md 에서 자동으로 뽑아낸 것이므로, 고칠 일이 있으면
그 페이지를 고치고 다시 뽑아내는 편이 낫다.
"""

"""
MNIST 퍼짐 모델

MNIST 숫자 만들어 내기를 위한 퍼짐 모델의 온전한 짜기.
학부생을 위한 그럴듯한 그림 만들어 내기 보기를 준다.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# ========================================================================
# 메인
# ========================================================================

from diffusion_utils import (
    cosine_beta_schedule,
    get_diffusion_parameters,
    get_loss,
    sample,
    visualize_samples
)
from unet_architecture import SimpleUNet


class MNISTDiffusion:
    """
    MNIST 퍼짐 모델의 익히기와 뽑기를 감싸는 갈래.
    """
    
    def __init__(self, 
                 timesteps: int = 1000,
                 batch_size: int = 128,
                 learning_rate: float = 2e-4,
                 device: str = None):
        """
        인수:
            timesteps: 퍼짐 때 걸음 수
            batch_size: 익히기 묶음 크기
            learning_rate: 최적화기의 학습률
            device: 익힐 장치('cuda' 또는 'cpu')
        """
        self.timesteps = timesteps
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # 퍼짐 매개변수를 채비한다
        betas = cosine_beta_schedule(timesteps)
        self.diffusion_params = get_diffusion_parameters(betas)
        
        # 매개변수를 기기로 옮긴다
        for key in self.diffusion_params:
            self.diffusion_params[key] = self.diffusion_params[key].to(self.device)
        
        # 모형을 시작한다
        self.model = SimpleUNet(
            in_channels=1,
            out_channels=1,
            base_channels=64,
            time_emb_dim=256
        ).to(self.device)
        
        # 최적화기
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate
        )
        
        # 더 나은 표본을 위한 지수 이동 평균 모델
        self.ema_model = SimpleUNet(
            in_channels=1,
            out_channels=1,
            base_channels=64,
            time_emb_dim=256
        ).to(self.device)
        self.ema_model.load_state_dict(self.model.state_dict())
        self.ema_decay = 0.9999
        
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def update_ema(self):
        """모델 매개변수의 지수 이동 평균을 고친다."""
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), 
                                        self.model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
    
    def get_dataloader(self, train: bool = True):
        """
        MNIST 자료 불러오개를 만든다.
        
        인수:
            train: 익히기 묶음을 불러올지 시험 묶음을 불러올지
        
        반환값:
            MNIST용 DataLoader
        """
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # [-1, 1]로 정규화
        ])
        
        dataset = datasets.MNIST(
            root='./data',
            train=train,
            download=True,
            transform=transform
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=train,
            num_workers=2,
            pin_memory=True
        )
    
    def train_epoch(self, dataloader):
        """한 에폭을 학습한다."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc="Training")
        for images, _ in pbar:
            images = images.to(self.device)
            
            # 아무 때 걸음을 뽑는다
            t = torch.randint(
                0, self.timesteps,
                (images.shape[0],),
                device=self.device
            )
            
            # 손실을 계산한다
            loss = get_loss(self.model, images, t, self.diffusion_params)
            
            # 최적화
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # 지수 이동 평균을 고친다
            self.update_ema()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def sample_images(self, n_samples: int = 64, use_ema: bool = True):
        """
        표본 그림을 만든다.
        
        인수:
            n_samples: 만들 표본의 개수
            use_ema: 지수 이동 평균 모델을 쓸지 여부
        
        반환값:
            만든 그림 텐서
        """
        model = self.ema_model if use_ema else self.model
        
        samples = sample(
            model,
            shape=(n_samples, 1, 28, 28),
            timesteps=self.timesteps,
            diffusion_params=self.diffusion_params,
            device=self.device
        )
        
        return samples
    
    def train(self, epochs: int, save_interval: int = 10):
        """
        퍼짐 모델을 익힌다.
        
        인수:
            epochs: 학습 에포크 수
            save_interval: N바퀴마다 표본을 갈무리한다
        """
        dataloader = self.get_dataloader(train=True)
        
        # 표본을 담을 자리를 만든다
        os.makedirs('samples', exist_ok=True)
        
        losses = []
        
        print(f"\nTraining for {epochs} epochs...")
        print("=" * 50)
        
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            
            # 학습
            avg_loss = self.train_epoch(dataloader)
            losses.append(avg_loss)
            
            print(f"Average loss: {avg_loss:.4f}")
            
            # 표본을 만들어 갈무리한다
            if epoch % save_interval == 0 or epoch == 1:
                print("Generating samples...")
                samples = self.sample_images(n_samples=64)
                visualize_samples(
                    samples,
                    nrow=8,
                    filename=f'samples/epoch_{epoch:04d}.png'
                )
            
            # 검사점 저장
            if epoch % 50 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')
        
        # 학습 손실 그리기
        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('MNIST Diffusion Training Loss')
        plt.grid(True)
        plt.savefig('training_loss.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("\nSaved training loss plot to training_loss.png")
        
        print("\n" + "=" * 50)
        print("Training complete!")
        print("=" * 50)
    
    def save_checkpoint(self, filename: str):
        """모델 되짚기 지점을 갈무리한다."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'ema_model_state_dict': self.ema_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'diffusion_params': self.diffusion_params,
        }, filename)
        print(f"Saved checkpoint to {filename}")
    
    def load_checkpoint(self, filename: str):
        """모델 되짚기 지점을 불러온다."""
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded checkpoint from {filename}")


def main():
    """
    MNIST 퍼짐의 으뜸 익히기 대본.
    """
    print("=" * 50)
    print("MNIST Diffusion Model Training")
    print("=" * 50)
    
    # 퍼짐 모델을 첫자리매김한다
    mnist_diffusion = MNISTDiffusion(
        timesteps=1000,
        batch_size=128,
        learning_rate=2e-4
    )
    
    # 학습
    mnist_diffusion.train(epochs=100, save_interval=10)
    
    # 마지막 표본을 만든다
    print("\nGenerating final samples...")
    samples = mnist_diffusion.sample_images(n_samples=64)
    visualize_samples(samples, nrow=8, filename='final_samples.png')
    
    # 마지막 모델을 저장한다
    mnist_diffusion.save_checkpoint('mnist_diffusion_final.pt')
    
    print("\nAll done! Check the 'samples' folder for generated images.")


if __name__ == "__main__":
    main()
