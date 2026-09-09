# MNIST 퍼짐 모델

이 단원은 요즘 만들어 내는 모델의 핵심 부품인 mnist 퍼짐 모델을 짠다. 여기서 보이는 개념과 재주를 알면 퍼짐 모델과 점수 바탕 만들어 내는 방법을 다루는 데 꼭 필요한 앎을 얻는다. 이 짜기는 또렷함과 실제 쓸모의 균형을 맞추어 배우기에도 실험하기에도 알맞다.

## 1. 코드

```python
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
```

## 2. 논의

mnist 퍼짐 모델의 짜기는 이 마당에 자리 잡은 방식을 따른다. 코드 짜임이 모델 뜻매김과 익히기 논리를 갈라 놓아 부품을 하나씩 고치기 쉽다. 얼개 고르기는 만들어 내는 모델 무리가 많은 실험에서 얻은 배움을 담고 있다.

이 짜기의 핵심에는 수치의 안정을 꼼꼼히 다루기, 고르게 맞추기 재주를 제대로 쓰기, 효율 좋은 셈 결이 든다. 익히기 절차에는 잡음 차례표, 기울기 다루기, 이따금의 따지기가 들며 모두 품질 높은 결과를 내는 데 결정적이다.

이 단원은 이론의 개념이 실제 짜기로 어떻게 옮겨지는지 보이며 만들어 내는 모델의 더 넓은 틀과 이어진다. 여기서 보이는 재주는 만들어 내는 모델이 이룰 수 있는 것의 가장자리를 넓히는 더 앞선 변형과 넓힘을 이해하는 바탕이 된다.

## 연습문제

**연습문제 1.**
구체적인 자료 묶음으로 이 단원의 으뜸 셈을 좇아라. 큰 걸음마다 텐서 꼴을 적고 모든 차원이 서로 맞는지 확인하라.

??? success "연습문제 1 풀이"
    모델에 알맞은 꼴의 들임 묶음에서 시작한다. 층이나 함수 부르기마다 셈을 따라가며 바뀜 뒤 텐서 꼴을 적는다. 겹말기 층에서는 내놓기 차원 공식을 쓴다. 눈길 얼개에서는 물음, 열쇠, 값의 차원이 맞는지 확인한다. 마지막 내놓기 꼴이 바라던 목표 차원과 맞는지 굳힌다. 이 익힘은 자료가 얼개를 어떻게 흐르는지에 대한 직관을 쌓아 준다.

---

**연습문제 2.**
이 단원에 쓰인 손실 함수를 가려내고 모델 매개변수에 대한 기울기를 이끌어 내라. 왜 이 손실 함수가 이 일에 알맞은지 설명하라.

??? success "연습문제 2 풀이"
    손실 함수는 모델이 헤아린 값과 목표 사이의 어긋남을 잰다. 잡음 헤아리기에서는 평균 제곱 어긋남 손실 $\|\epsilon - \epsilon_\theta(x_t, t)\|^2$을 쓰는데, 이것이 로그 가능도의 변분 아래 한계에 맞물리기 때문이다. 매개변수 $\theta$에 대한 기울기는 $-2(\epsilon - \epsilon_\theta) \nabla_\theta \epsilon_\theta$이며 헤아림 어긋남을 줄이는 방향을 가리킨다. 이 손실을 가장 작게 하는 것이 퍼짐 모델에서 자료 로그 가능도의 아래 한계를 가장 크게 하는 것과 같으므로 알맞다.

---

**연습문제 3.**
다른 잡음 차례표를 받쳐 주도록 이 짜기를 고쳐라(예컨대 선형에서 코사인으로, 또는 그 반대로). 두 차례표의 익히기 움직임과 표본 품질을 견주어라.

??? success "연습문제 3 풀이"
    두 차례표를 모두 짜고 각각으로 모델을 익힌다. $\bar{\alpha}_t = \cos^2\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)$으로 뜻매김한 코사인 차례표는 선형 차례표 $\beta_t = \beta_{\min} + t(\beta_{\max} - \beta_{\min})/T$에 견주어 잡음이 더 매끄럽게 늘어난다. 손실 곡선을 좇고 일정한 사이마다 표본을 만든다. 코사인 차례표는 신호 대 잡음비가 더 완만하게 줄어 때 걸음에 걸쳐 배움 신호가 더 고르므로 흔히 더 좋은 결과를 낸다.

## 정리하며

**다룬 것** — MNIST 퍼짐 모델

mnist 퍼짐 모델의 짜기는 이 마당에 자리 잡은 방식을 따른다.

고갱이 갈래는 `MNISTDiffusion`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
